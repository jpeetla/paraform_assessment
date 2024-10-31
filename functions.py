import os
from bs4 import BeautifulSoup
from pinecone import Pinecone
import re
from langchain_huggingface import HuggingFaceEmbeddings
import webbrowser
import data_analysis
from openai import OpenAI
import concurrent.futures


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("paraform")
categoryembeddings_index = pc.Index("categoryembeddings")

categories_list = [
    "Software Engineering", 
    "Product & Project Management", 
    "Data Science & Machine Learning", 
    "Business Operations & Strategy", 
    "Marketing & Growth", 
    "Design & User Experience (UX/UI)", 
    "Customer Success & Sales", 
    "Operations & Support", 
    "Engineering Management & Leadership", 
    "Specialized Engineering",
    "Dev-Ops"
]

def clean_html(html):
    if not html or not isinstance(html, str):  # Check if html is a non-empty string
        return ''
    
    if re.search(r'<[^>]+>', html):
        return BeautifulSoup(html, 'html.parser').get_text(separator=" ").strip()
    
    return html.strip()

def embed_roles(embeddings, roles_context):
    role_embeddings = {}
    for key, value in roles_context.items():
        query_result = embeddings.embed_query(value)
        role_embeddings[key] = query_result
    return role_embeddings

def upload_pinecone(roles_dict):
    i = 0
    for key in roles_dict.keys():
        years_experience_min = roles_dict[key]['years_experience_min']
        if years_experience_min == None:
            years_experience_min = 0

        role_category = data_analysis.job_category_mapping(roles_dict[key]['name'])
        index.update(
	        id=key, 
	        set_metadata={"years_experience_min": years_experience_min, "role_category": role_category} 
            )
        print(i)
        i += 1

def calc_candidate_experience_years(candidates_dict, cand_id):
    end = int(candidates_dict[cand_id]['candidate']['experiences'][0]['start_date'][:4])  
    first_experience = len(candidates_dict[cand_id]['candidate']['experiences']) - 1
    start = int(candidates_dict[cand_id]['candidate']['experiences'][first_experience]['start_date'][:4])
    return end - start

def create_candidate_embedding(embeddings, candidates_dict, candidate_id):
    experience = ""
    for exp in candidates_dict[candidate_id]['candidate']['experiences']:
        r_t, d = "", ""
        if type(exp['role_title']) == str:
            r_t = exp['role_title']
        if type(exp['description']) == str:
            d = exp['description']

        summary =  r_t + d
        experience += summary

    # embedding = embeddings.embed_query(experience)
    # category = categorize_candidate_with_llama(experience, categories_list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the embedding and categorization tasks
        embedding_future = executor.submit(embeddings.embed_query, experience)
        category_future = executor.submit(categorize_candidate_with_llama, experience, categories_list)

        # Retrieve the results from each future
        embedding = embedding_future.result()
        category = category_future.result()
        
    print("Candidate Category:")
    print(category)

    candidate_experience_years = calc_candidate_experience_years(candidates_dict, candidate_id)
    return category, embedding, candidate_experience_years

def categorize_candidate_with_llama(profile_description, categories):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""I will give you a description of a candidate and his/her experience. Please categorize the candidate into one of the following categories: Software Engineering, Product & Project Management, 
                            Data Science & Machine Learning, Business Operations & Strategy, Marketing & Growth, Design & User Experience (UX/UI), Customer Success & Sales, Operations & Support, Engineering Management & Leadership, 
                            Specialized Engineering, Dev-Ops. Here is the candidate description: {profile_description}. Say nothing else besides the assigned category. """
            }
        ]
    )

    return completion.choices[0].message.content

def query_roles(category, candidate_embedding, candidate_experience_years):
    # category_response = categoryembeddings_index.query(
    #     vector=candidate_embedding,
    #     top_k=2
    # )

    # category_response_dict = category_response.to_dict()
    # categories = [obj["id"] for obj in category_response_dict["matches"]]
    # print(categories)

    response = index.query(
        vector=candidate_embedding,
        top_k=3,
        include_values=False,
        include_metadata=True,
        filter={"years_experience_min": {"$lte": candidate_experience_years},
                "role_category": {"$in": [category]}}
    )
    
    return response

def open_linkedin(username):
    url = "https://www.linkedin.com/in/{}".format(username)
    webbrowser.open(url)

def open_company_profile(role_id):
    url = "https://www.paraform.com/company/melange/{}".format(role_id)
    print(url)
    webbrowser.open(url)