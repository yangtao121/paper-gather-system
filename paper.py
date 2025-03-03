from core.dify import DifyClient
from core.searngx import SearxTool
from loguru import logger
import yaml
import time


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

    def searchPaperByTopic(self, topic: str,
                           search_limit: int = 10,
                           searngx_kwargs: dict = {}
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

    paper_processor.searchPaperByTopic(topic="learning navigation", search_limit=10)
