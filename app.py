import json
from flask import Flask, request, jsonify
import functions
from langchain_huggingface import HuggingFaceEmbeddings
import os

app = Flask(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

roles_file_path = 'roles.json'
candidates_file_path = 'candidates.json'
with open(roles_file_path, 'r') as file:
    roles_json = json.load(file)

with open(candidates_file_path, 'r') as file:
    candidates_json = json.load(file)

#Convert to Dictionary
candidates_dict = {entry['id']: {k: v for k,  v in entry.items() if k != 'ID'} for entry in candidates_json}
roles_dict = {entry['id']: {k: v for k, v in entry.items() if k != 'ID'} for entry in roles_json}

# # #Create Context Paragraph for each Role
# roles_context = {} #(ID, Context)
# for key, value in roles_dict.items():
#     requirements_text = ' '.join(
#         functions.clean_html(item['description']) + ' ' + functions.clean_html(item.get('explanation', ''))
#         for item in value['requirements']
#     )
#     combined_text = f"{requirements_text} {value['responsibilities']} {value['tech_stack']}"
#     roles_context[key] = combined_text

# # role_embeddings = functions.embed_roles(roles_context) #(ID, Embedding)
# functions.upload_pinecone(roles_dict)

# Candidate Query
@app.route('/query_roles', methods=['GET'])
def query_roles():
    candidate_id = request.args.get("candidate_id")
    candidate_data = functions.get_candidate_info(candidate_id)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if not candidate_id:
        return jsonify({"error": "candidate_id is required"}), 400

    category, candidate_embedding, candidate_years_experience = functions.create_candidate_embedding(embeddings, candidate_data, candidate_id)
    result = functions.query_roles(category, candidate_embedding, candidate_years_experience)
    response_dict = result.to_dict()

    candidate_linkedin = candidate_data['linkedin_user']
    functions.open_linkedin(candidate_linkedin)
    ids = [obj["id"] for obj in response_dict["matches"]]
    for role_id in ids:
        functions.open_company_profile(role_id)
    
    return jsonify(ids)

if __name__ == '__main__':
    app.run(debug=False)


