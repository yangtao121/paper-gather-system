from core.dify import DifyClient
from core.searngx import SearxTool
from core.paperless import PaperlessClient

from loguru import logger

import yaml
import time
import os


class PaperProcessor:
    def __init__(self,
                 config: str,
                 ):
        """
        用于处理论文的功能类。

        :param config: 配置文件路径
        :type config: str
        """
        with open(config, "r") as f:
            self.config = yaml.safe_load(f)

        self.dify_client = DifyClient(
            base_url=self.config["dify_url"],
        )

        self.searngx_client = SearxTool(
            search_host=self.config["searxng_host"]
        )

        self.paperless_client = PaperlessClient(
            base_url=self.config["paperless_url"],
            auth=self.config["paperless_token"]
        )

        # 文件系统，每个模块只管理到自己能触及的层面
        self.cache_dir = self.config["cache_dir"]

        # 检查缓存区有没有创建
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def searchPaperByTopic(self, topic: str,
                           search_limit: int = 10,
                           check_exist: bool = True,
                           searngx_kwargs: dict = {},

                           ):
        """
        通过主题搜索相应的论文
        """
        start_time = time.time()

        search_result = self.searngx_client.arxivSearch(query=topic,
                                                        num_results=search_limit,
                                                        kwargs=searngx_kwargs
                                                        )

        end_time = time.time()

        logger.info(
            f"Successfully search {search_result.num_results} papers in {end_time - start_time} seconds.")
        logger.info("Start to analyze the papers...")

        llm_total_start_time = time.time()

        for paper in search_result:
            llm_start_time = time.time()
            # check the paper has been analyzed
            if check_exist and self.paperless_client.document_exists(paper.title):
                logger.info(
                    f"The paper \"{paper.title}\" has been analyzed.")
                continue
            logger.info(
                f"Start to check the paper \"{paper.title}\" corresponding to the topic \"{topic}\".")
            result = self.dify_client.workflow_run(
                inputs={
                    "title": paper.title,
                    "snippet": paper.snippet,
                    "theme": topic,
                },
                api_key=self.config["dify_api_paper_detect"],
                user=self.config["dify_user"]
            )
            

            llm_eval = eval(result['data']['outputs']['result'])
            logger.info(f"The rating of the paper is {llm_eval['rating']}.")
            logger.info(
                f"The justification of the paper is {llm_eval['justification']}.")

            # start to analyze the paper tag

            if llm_eval['rating'] == "A" or llm_eval['rating'] == "B":
                # 创建该文档工作空间
                paper_workspace = os.path.join(
                    self.config["cache_dir"], paper.clear_invalid_characters(paper.title))
                if not os.path.exists(paper_workspace):
                    os.makedirs(paper_workspace)

                # 分析该文档的标签
                result = self.dify_client.workflow_run(
                    inputs={
                        "title": paper.title,
                        "abstract": paper.snippet,
                        # "theme": topic,
                    },
                    api_key=self.config["dify_api_paper_tag"],
                    user=self.config["dify_user"]
                )

                llm_eval = eval(result['data']['outputs']['result'])
                logger.info(f"The tag of the paper is {llm_eval['keywords']}.")

                paper.setTag(llm_eval['keywords'])

                # download the paper
                logger.info(
                    f"Start to download the paper \"{paper.title}\".")

                if self.config["save_paper"]:
                    paper.downloadPdf(
                        save_path=paper_workspace,
                    )

                    paperless_result = self.paperless_client.upload_document(
                        file_path=paper.pdf_path,
                        title=paper.title,
                        tag_names=paper.tag,
                        auto_create_tags=True,
                    )

                    # print(paperless_result)
                    logger.info(
                        f"Get response from paperless: {paperless_result}")

                    # 删除缓存
                    paper.clearPdf()
                    # os.remove(paper.pdf_path)

            llm_end_time = time.time()
            logger.info(
                f"Successfully analyze the paper \"{paper.title}\" in {llm_end_time - llm_start_time} seconds.")

        llm_total_end_time = time.time()
        logger.info(
            f"Successfully analyze {search_result.num_results} papers in {llm_total_end_time - llm_total_start_time} seconds.")


if __name__ == "__main__":
    paper_processor = PaperProcessor(
        config="config/paper.yaml"
    )

    paper_processor.searchPaperByTopic(
        topic="LLM navigation", search_limit=1,
        check_exist=False
    )
