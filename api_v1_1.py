import os

from argparse import ArgumentParser
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import BingSearch
from knowledge_storm.utils import load_api_key
from knowledge_storm.storm_wiki import BaseCallbackHandler
from knowledge_storm.storm_wiki import StormInformationTable, StormArticle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import logging
import time

# 配置日志记录
logging.basicConfig(
    filename='api.log',         # 日志文件名
    filemode='a',                   # 追加模式 ('w' 为覆盖模式)
    level=logging.DEBUG,            # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

app = FastAPI()

load_api_key(toml_file_path='secrets.toml')

parser = ArgumentParser()

parser.add_argument(
    '--server-name',
    type=str,
    required=True,
)

parser.add_argument(
    '--port',
    type=int,
    required=True,
)


parser.add_argument(
    '--max-thread-num',
    type=int,
    default=3,
    help='Maximum number of threads to use. The information seeking part and the article generation'
    'part can speed up by using multiple threads. Consider reducing it if keep getting '
    '"Exceed rate limit" error when calling LM API.',
)

parser.add_argument(
    '--max-conv-turn',
    type=int,
    default=3,
    help='Maximum number of questions in conversational question asking.',
)

parser.add_argument(
    '--max-perspective',
    type=int,
    default=3,
    help='Maximum number of perspectives to consider in perspective-guided question asking.',
)
parser.add_argument(
    '--search-top-k',
    type=int,
    default=3,
    help='Top k search results to consider for each search query.',
)

args = parser.parse_args()

lm_configs = STORMWikiLMConfigs()
openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'api_base': os.getenv('OPENAI_API_BASE'),
    'temperature': 1.0,
    'top_p': 0.9,
}


gpt_35_model_name = 'gpt-4o-mini-2024-07-18'
gpt_4_model_name = 'gpt-4o-mini-2024-07-18'

conv_simulator_lm = OpenAIModel(
    model=gpt_35_model_name, max_tokens=500, **openai_kwargs
)
question_asker_lm = OpenAIModel(
    model=gpt_35_model_name, max_tokens=500, **openai_kwargs
)
outline_gen_lm = OpenAIModel(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
article_gen_lm = OpenAIModel(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
article_polish_lm = OpenAIModel(
    model=gpt_4_model_name, max_tokens=4000, **openai_kwargs
)

lm_configs.set_conv_simulator_lm(conv_simulator_lm)
lm_configs.set_question_asker_lm(question_asker_lm)
lm_configs.set_outline_gen_lm(outline_gen_lm)
lm_configs.set_article_gen_lm(article_gen_lm)
lm_configs.set_article_polish_lm(article_polish_lm)

engine_args = STORMWikiRunnerArguments(
    # output_dir=args.output_dir,
    max_conv_turn=args.max_conv_turn,
    max_perspective=args.max_perspective,
    search_top_k=args.search_top_k,
    max_thread_num=args.max_thread_num,
)
rm = BingSearch(
    bing_search_api=os.getenv('BING_SEARCH_API_KEY'),
    k=engine_args.search_top_k,
)

def predict(topic):
    def yield_func(data):
        """
        内部函数，用于将数据通过 yield 返回给前端。
        """
        yield_data.append(data) 
    yield_data = []  

    runner = STORMWikiRunner(engine_args, lm_configs, rm)
    runner.topic = topic
    callback_handler = StreamlitCallbackHandler(yield_func=yield_func)
    # step1 do research
    information_table: StormInformationTable = None
    conversation_log, information_table = runner.run_knowledge_curation_module(
        ground_truth_url=None, callback_handler=callback_handler
    )

    url_dict = information_table.dump_url_to_dict()
    response_data = json.dumps({"type": "url", "data": url_dict}) + "\n"
    logging.info(list(url_dict.keys()))
    logging.info("="*100)
    yield f"data: {response_data}\n\n"
 
    # step2 do_generate_outline
    outline = runner.run_outline_generation_module(
        information_table=information_table, callback_handler=callback_handler
    )
    outline_str = outline.dump_outline_to_str()
    response_data = json.dumps({"type": "outline", "data": outline_str}) + "\n"
    logging.info([response_data])
    logging.info("="*100)
    yield f"data: {response_data}\n\n"


    # step3 do_generate_article
    draft_article: StormArticle = None
    draft_article = runner.run_article_generation_module(
        outline=outline,
        information_table=information_table,
        callback_handler=callback_handler,
    )

    # step4 do_polish_article
    polished_article = runner.run_article_polishing_module(
        draft_article=draft_article, remove_duplicate=True
    )
    polished_article = polished_article.to_string()
    response_data = json.dumps({"type": "article", "data": polished_article}) + "\n"
    logging.info([response_data])
    logging.info("="*100)
    yield f"data: {response_data}\n\n"


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, yield_func):
        """
        初始化回调处理器。
        :param yield_func: 一个将数据发送给后端的函数，比如 yield。
        """
        self.yield_func = yield_func

    def on_dialogue_turn_end(self, dlg_turn):
        """
        当对话回合结束时被调用。
        :param dlg_turn: 本次对话回合的数据。
        """
        urls = list(set([r.url for r in dlg_turn.search_results]))
        dlg_turn_data = json.dumps({"type": "dlg_turn", "data": urls}) + "\n"
        self.yield_func(f"data: {dlg_turn_data}\n\n")



class ArticleRequestItem(BaseModel):
    topic: str


@app.post("/generate")
def generate_article(articleRequestItem: ArticleRequestItem):
    topic = articleRequestItem.topic
    logging.info(f"Received request with topic: {topic}")
    
    try:
        response = StreamingResponse(predict(topic), media_type="text/event-stream")
        logging.info(f"Streaming response for topic: {topic}")
        return response
    except Exception as e:
        logging.error(f"Error while generating article for topic '{topic}': {e}")
        raise


if __name__ == '__main__':

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_keep_alive=120)

