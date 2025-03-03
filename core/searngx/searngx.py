from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools.searx_search.tool import SearxSearchResults
from loguru import logger
import pprint


import requests


class ArxivData:
    def __init__(self, result: dict):
        """
        用于存储单条arxiv 的搜索结果。
        输入的 result 必须包含的 key 如下：
        - title: 标题
        - link: 链接
        - snippet: 摘要
        - categories: 分类
        :param result: 单条搜索结果
        :type result: dict
        """
        self.title = None
        self.link = None
        self.snippet = None
        self.categories = None

        for key, value in result.items():
            setattr(self, key, value)

        # 获取pdf链接
        self.pdf_link = self.link.replace("abs", "pdf")

        # 论文的tag
        self.tag = None

    def setTag(self, tag: str):
        """
        设置论文的tag
        """
        self.tag = tag

    # def downloadPdf(self, save_path: str):
    #     """
    #     下载pdf
    #     """
    #     response = requests.get(self.pdf_link)
    #     file_name =
    #     with open(save_path, "wb") as f:
    #         f.write(response.content)


class ArxivResult:

    def __init__(self, results: list[dict]):
        """
        搜索结果的保存类。

        :param results: 搜索结果
        :type results: list[dict]
        """
        self.results = [ArxivData(result) for result in results]

        self.num_results = len(self.results)

    def __iter__(self):
        """
        实现迭代器协议
        """
        return iter(self.results)


class SearxTool:
    def __init__(self, search_host: str):
        """
        用于调用 searxng 的 api，并简化返回结果，提供给大模型使用。

        :param search_host: searxng 的 host
        :type search_host: str
        """
        self.search_host = search_host
        self.search_wrapper = SearxSearchWrapper(searx_host=search_host)

    def arxivSearch(self, query: str,
                    num_results: int = 5,
                    kwargs: dict = {
                        "engines": ["arxiv"],
                    }
                    ) -> ArxivResult:
        """
        用于搜索 arxiv 的 api，并将结果转换为list[dict]。

        :param query: 搜索的查询
        :type query: str
        :param num_results: 返回的结果数量
        :type num_results: int
        :return: 搜索结果
        :rtype: ArxivResult
        """

        default_kwargs = {
            "engines": ["arxiv"],
        }

        default_kwargs.update(kwargs)
        arxiv_tool = SearxSearchResults(name="Arxiv", wrapper=self.search_wrapper,
                                        num_results=num_results,
                                        kwargs=default_kwargs)

        results = arxiv_tool.invoke(query)

        eval_results = eval(results)

        if not self.checkResult(eval_results):
            logger.error(f"No good Search Result, please try again.")
            return ArxivResult([])

        logger.info(f"Successfully get the result from arxiv.")
        return ArxivResult(eval_results)

    def checkResult(self, results: list[dict]) -> bool:
        """
        检查搜索结果是否为空。
        """

        if 'Result' in results[0]:
            return False
        return True


if __name__ == "__main__":
    searngx_tool = SearxTool(search_host="http://192.168.5.54:8080")
    results = searngx_tool.arxivSearch(query="",
                                       num_results=5)
    print(results.results[0].title)
    # print(results.results[0].link)
    print(results.results[0].snippet)
