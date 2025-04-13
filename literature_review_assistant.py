from anaconda_navigator.api.external_apps.bundle.installers import retrieve_and_validate
from google import genai
from google.genai import types

import json

import os
from dotenv import load_dotenv

import pandas as pd

import requests
import xml.etree.ElementTree as ET
import datetime

from chromadb import Documents, EmbeddingFunction, Embeddings
# from google.api_core import retry
import chromadb
from nltk.corpus.reader import documents


from numpy.array_api import result_type
from starlette.routing import request_response

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)



def get_model_response(prompt: str) -> str:
    config = types.GenerateContentConfig(temperature=0.0)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=config,
        contents=prompt,
    )

    return response.text


def generate_best_query(request: str, main_query=False) -> list:
    prompt = f'''You are a helpful research assistant doing a literature review. The researcher says: "{request}". What would be the most accurate arXiv API call to find this information? Please provide the API call alone, no need for an explanation. 
        INSTRUCTIONS: 
        Step 1 - consider how an arXiv API call is constructed. 
        Step 2 - define the best search query to use for the researcher's request
        Step 3 - construct the arXiv API call
        Step 4 - make sure the arXiv API call is valid, and fix it if it's not'''

    response = get_model_response(prompt)

    best_query = response.strip().replace("\n", "")
    if main_query:
        best_query_pages = [best_query, best_query.replace("start=0", "start=10"),
                            best_query.replace("start=0", "start=20")]

        return best_query_pages

    return best_query


def generate_subtopic_query(request: str) -> list:
    prompt = f'''You are a helpful research assistant doing a literature review. The researcher says: "{request}". What would be {number_of_subtopics} relevant sub-topics to gain a better understanding of this matter? Please provide a list of these topics. Provide no explanation. Return a Python list. '''

    close_topics = get_model_response(prompt)
    json_start = close_topics.find("[")
    s = close_topics[json_start:].replace("`", "").replace("[", "").replace("]", "").replace('"', "").replace("\n", "")

    subtopics = [i.strip() for i in s.split(", ")]

    subtopic_queries = []

    for topic in subtopics:
        subtopic_query = f"I want to look into {topic}"
        r = generate_best_query(subtopic_query)
        subtopic_queries.append(r.strip().replace("\n", ""))

    return subtopics, subtopic_queries

def generate_search_queries(request: str) -> list:
    main_query = generate_best_query(request, main_query=True)
    subtopic_results = generate_subtopic_query(request)
    subtopic_queries = subtopic_results[1]
    subtopic_list = subtopic_results[0]
    return main_query + subtopic_queries, subtopic_list


def get_arxiv_metadata(url: str, topic: str) -> tuple:
    arxiv_entries = {}

    url = url.replace(" ", "").replace("`", "")

    arxiv_response = requests.get(url)
    if arxiv_response.status_code != 200:
        raise requests.HTTPError(f"API call failed with status code {response.status_code}")
        return
    else:
        root = ET.fromstring(arxiv_response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            authors = [author.find('atom:name', ns).text.strip()
                       for author in entry.findall('atom:author', ns)]
            link = entry.find('atom:link', ns).attrib['href']
            id = entry.find('atom:id', ns).text.strip()
            updated = entry.find('atom:updated', ns).text.strip()
            updated = datetime.datetime.strptime(updated, '%Y-%m-%dT%H:%M:%SZ').date()
            if id not in arxiv_entries:
                arxiv_entries[id] = {
                    'summary': summary,
                    'metadatas': {
                        'authors': ", ".join(authors),
                        'title': title,
                        'published_url': link,
                        'updated': str(updated),
                        'id_url': id,
                        'original_search': url,
                        'topic': topic
                    }
                }

        return arxiv_entries



# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    # @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


DB_NAME = "arxiv_results"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)


def generate_db_content(call_list: list, subtopics: list):
    counter = 0
    subtopic_idx = 0
    n = len(subtopics)
    calls = len(call_list)
    for i in range(calls):
        if i > calls - n - 1:
            subtopic = subtopics[subtopic_idx]
            subtopic_idx += 1
        else:
            subtopic = "main"
        api_call = call_list[i]
        response = get_arxiv_metadata(api_call, subtopic)
        if response is not None:
            for article_id in response:
                db.add(documents=[response[article_id]["summary"]], metadatas=[response[article_id]["metadatas"]],
                       ids=[article_id])
                counter += 1

full_result = generate_search_queries(query)
r, st = full_result[0], full_result[1]

generate_db_content(r, st)


embed_fn.document_mode = False

result = db.query(query_texts=[query], n_results= 3 + 1)

[all_passages] = result["documents"]
all_titles = [j["title"] for i in result["metadatas"] for j in i]
all_authors = [j["authors"] for i in result["metadatas"] for j in i]
[all_ids] = result["ids"]


best_article = db.peek(1)

best_article = [best_article["metadatas"][0]["title"], best_article["metadatas"][0]["authors"], best_article["documents"], best_article["ids"]]

exclude_best = '0' in result["ids"][0]

if exclude_best:
    q = all_ids.index('0')
    # best_passage = all_passages[q]
    # best_title = all_titles[q]
    # best_metadata = result["metadatas"][q]
    all_titles = all_titles[:q] + all_titles[q + 1:]
    all_passages = all_passages[:q] + all_passages[q + 1:]
    all_ids = all_ids[:q] + all_ids[q + 1:]
    all_authors = all_authors[:q] + all_authors[q + 1:]



more_article_data = list(zip(all_titles, all_authors, all_passages, all_ids))


def get_subtopic_articles(subtopics: list, ids: list) -> list:
    subtopic_contents = []

    for subtopic in subtopics:
        results = db.get(
            where={"topic": subtopic},
            limit=10)

        idx = None
        subtopic_data = None
        for i in results["ids"]:
            if i not in ids:
                idx = results["ids"].index(i)
                break

        if idx is not None:
            article_content = results["documents"][idx]
            article_id = results["ids"][idx]
            article_title = results["metadatas"][idx]["title"]
            article_authors = results["metadatas"][idx]["authors"]
            subtopic_data = article_title, article_authors, article_content, article_id

        if subtopic_data:
            subtopic_contents.append(subtopic_data)
    return subtopic_contents


def generate_summary(request: str, subtopics = st, ids = all_ids) -> str:
    prompt = f'''You are a helpful research assistant. The researcher says: "{request}". 
Here is the data on the main article on the topic:
 {best_article}
 Here is the data on a few more articles on the topic: 
 {more_article_data}
PLease provide a short summary for each article (a couple of sentences). Please highlight the main article, providing a summary of up to 4 sentences about it. 
For each article mentioned, be sure to include the article title, authors, and id.  
Please don't explain that you got it. 
'''

    response = get_model_response(prompt)
    if not subtopics:
        return response
    else:
        subtopic_string = "\n".join(subtopics)
        keyword_text = (f"\n\n\n----some helpful keywords/topics may be: {subtopic_string}")

        subtopic_articles = get_subtopic_articles(subtopics, ids=ids)
        subtopics_prompt = f'''You are a helpful research assistant. The researcher says: "{request}". 
Here are some articles that another assistant suggested to further explore this matter: 
{subtopic_articles}

Please provide a short summary of how the things discussed in the articles add to the concept the researcher was talking about. 
For each article mentioned, be sure to include the article title, authors, and id.  
If no article is relevant, simply return " ".
Please don't explain that you got it. 
'''
        subtopics_response = get_model_response(subtopics_prompt)
        return response + keyword_text + "\n\n\n" + subtopics_response
