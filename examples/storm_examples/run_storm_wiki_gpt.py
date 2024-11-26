import os

from argparse import ArgumentParser
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
    AzureAISearch,
)
from knowledge_storm.utils import load_api_key


def main(args):
    load_api_key(toml_file_path='secrets.toml')
    lm_configs = STORMWikiLMConfigs()
    openai_kwargs = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'api_base': os.getenv('OPENAI_API_BASE'),
        'temperature': 1.0,
        'top_p': 0.9,
    }

    ModelClass = OpenAIModel
    gpt_35_model_name = 'gpt-4o-mini-2024-07-18'
    gpt_4_model_name = 'gpt-4o-mini-2024-07-18'

    conv_simulator_lm = ModelClass(
        model=gpt_35_model_name, max_tokens=500, **openai_kwargs
    )
    question_asker_lm = ModelClass(
        model=gpt_35_model_name, max_tokens=500, **openai_kwargs
    )
    outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
    article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
    article_polish_lm = ModelClass(
        model=gpt_4_model_name, max_tokens=4000, **openai_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.

    match args.retriever:
        case 'bing':
            rm = BingSearch(
                bing_search_api=os.getenv('BING_SEARCH_API_KEY'),
                k=engine_args.search_top_k,
            )
        case 'you':
            rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)
        case 'brave':
            rm = BraveRM(
                brave_search_api_key=os.getenv('BRAVE_API_KEY'),
                k=engine_args.search_top_k,
            )
        case 'duckduckgo':
            rm = DuckDuckGoSearchRM(
                k=engine_args.search_top_k, safe_search='On', region='us-en'
            )
        case 'serper':
            rm = SerperRM(
                serper_search_api_key=os.getenv('SERPER_API_KEY'),
                query_params={'autocorrect': True, 'num': 10, 'page': 1},
            )
        case 'tavily':
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv('TAVILY_API_KEY'),
                k=engine_args.search_top_k,
                include_raw_content=True,
            )
        case 'searxng':
            rm = SearXNG(
                searxng_api_key=os.getenv('SEARXNG_API_KEY'), k=engine_args.search_top_k
            )
        case 'azure_ai_search':
            rm = AzureAISearch(
                azure_ai_search_api_key=os.getenv('AZURE_AI_SEARCH_API_KEY'),
                k=engine_args.search_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", "searxng", or "azure_ai_search"'
            )

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input('Topic: ')
    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


if __name__ == '__main__':
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/gpt',
        help='Directory to store the outputs.',
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
        '--retriever',
        type=str,
        choices=[
            'bing',
            'you',
            'brave',
            'serper',
            'duckduckgo',
            'tavily',
            'searxng',
            'azure_ai_search',
        ],
        help='The search engine API to use for retrieving information.',
    )
    # stage of the pipeline
    parser.add_argument(
        '--do-research',
        action='store_true',
        help='If True, simulate conversation to research the topic; otherwise, load the results.',
    )
    parser.add_argument(
        '--do-generate-outline',
        action='store_true',
        help='If True, generate an outline for the topic; otherwise, load the results.',
    )
    parser.add_argument(
        '--do-generate-article',
        action='store_true',
        help='If True, generate an article for the topic; otherwise, load the results.',
    )
    parser.add_argument(
        '--do-polish-article',
        action='store_true',
        help='If True, polish the article by adding a summarization section and (optionally) removing '
        'duplicate content.',
    )
    # hyperparameters for the pre-writing stage
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
    # hyperparameters for the writing stage
    parser.add_argument(
        '--retrieve-top-k',
        type=int,
        default=3,
        help='Top k collected references for each section title.',
    )
    parser.add_argument(
        '--remove-duplicate',
        action='store_true',
        help='If True, remove duplicate content from the article.',
    )

    main(parser.parse_args())
