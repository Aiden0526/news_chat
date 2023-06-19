import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from newsdataapi import NewsDataApiClient
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from retrying import retry
import argparse
import json

from langdetect import detect
# from google.cloud import translate_v2 as translate
# from translate import Translator
from googletrans import Translator


@retry(stop_max_attempt_number=3,wait_fixed=2000)
def get_news_content(keyword_lst):
    #Init

    newsapi = NewsDataApiClient(apikey='pub_24650587200147930ec207fb6873c479ce98a')

    formatted_keyword = [f'"{string}"' for string in keyword_lst]
    topic_keyword = ' AND '.join(formatted_keyword)
    # print(topic_keyword)
    response = newsapi.news_api(q=topic_keyword,
                                language="en")

    # print(response)

    # print(response)

    articles = response['results']
    contents = [(article['content'],article['source_id']) for article in articles]

    # print(contents)


    return contents


def question_to_keyword(ques):

    nlp = spacy.load("en_core_web_trf")
    text = nlp(ques)
    keywords = [chunk.text for chunk in text.noun_chunks]
    # print("keyword: ",keywords)
    return keywords

def similar_content_rank(ques,contents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ques_embedding = model.encode([ques])
    # print(contents)
    contents_embedding = model.encode(contents[0])

    if len(ques_embedding.shape)==1:
        ques_embedding = ques_embedding.reshape(1,-1)

    if len(contents_embedding.shape)==1:
        contents_embedding = contents_embedding.reshape(1,-1)


    sim_score = cosine_similarity(ques_embedding,contents_embedding)

    contents_score = zip(contents,sim_score[0])

    ranked_content = sorted(contents_score,key=lambda x:x[1], reverse=True)

    return ranked_content


@retry(stop_max_attempt_number=3,wait_fixed=2000)
def get_answer(most_relative_news,myprompt,ques):

    combined_prompt = f"{myprompt} \n\n###\n\n{most_relative_news} \n\n###\n\n{ques}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            max_tokens=1000,
            messages=[
                {"role":"user","content":combined_prompt}
            ]
        )

        ## save the tree of thought process to full_analysis.json file
        output_file = os.path.join(os.getcwd(), "full_analysis.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        steps = response['choices'][0]['message']['content'].split('Step')

        analysis = {}

        for i, step in enumerate(steps):
            step = step.strip().lstrip(':')
            analysis[f"Step {i+1}"] = step

        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=4)

        return post_process(response['choices'][0]['message']['content'])
    except Exception as e:
        print("There is an error: ",e)
        return None


def post_process(answer):
    steps = answer.split("Step ")

    for step in steps:
        if step.startswith("4:"):
            lines = step.split("\n")
            return "\n".join(lines[1:])

    return answer



def parse_arge():
    args = argparse.ArgumentParser()

    args.add_argument('--api_key',required=True,help='Enter your OpenAI API key')
    args.add_argument('--question',required=True,help='Enter the question that you want to ask')

    args = args.parse_args()
    return args

def detect_language(text):
    lang = detect(text)
    return lang

def translator_en_ch(text):
    translator = Translator()
    # print(text)
    translation = translator.translate(text,src='en',dest='zh-cn')
    # print(translation)

    return translation.text


if __name__ == '__main__':
    try:
        args = parse_arge()
        openai.api_key = args.api_key

        ques = args.question


        lang = detect_language(ques)
        # print(ques)
        if lang == 'zh-cn':
            translator = Translator()
            translation = translator.translate(ques,src='zh-cn',dest='en')
            # print(translation)
            ques = translation.text

        keyword_lst = question_to_keyword(ques)
        # print(keyword_lst)
        news_content = get_news_content(keyword_lst)
        # print(len(news_content[0]))

        prompt_file = 'myprompt.txt'
        with open(prompt_file, 'r') as f:
            myprompt = f.read()


        if news_content:
            rerank_content = similar_content_rank(ques,news_content[0])
            # print(rerank_content[0])

            if lang == 'en':
                # print("english")
                print("Question: ", ques)
                print("Answer: ",get_answer(rerank_content[0][0],myprompt,ques))
                print("Source: ",news_content[0][1])
            else:
                # print("chinese")
                print("问题: ", translator_en_ch(ques))
                eng_text = get_answer(rerank_content[0][0],myprompt,ques)
                # print(rerank_content[0][0])
                # print("eng text:",eng_text)
                answer = translator_en_ch(eng_text)
                if answer:
                    print("答案: ",answer)
                else:
                    print(get_answer(rerank_content[0][0],myprompt,ques))
                print("来源：", news_content[0][1])


        else:
            no_news = "No news found, Please search relevant news in your training data and treat them as the news"
            if lang == 'en':
                print(get_answer(no_news,myprompt,ques))

            else:
                answer = translator_en_ch(get_answer(no_news,myprompt,ques))
                print(answer)
    except Exception as e:
        print("There is an error: ",e)




