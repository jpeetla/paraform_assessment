import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("categoryembeddings")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

role_category_description = {
    "Software Engineering": (
        "Focuses on software design, development, and maintenance across backend, frontend, and full-stack roles. "
        "Involves building, testing, and deploying applications or systems, often for distributed and scalable architectures. "
        "Includes specialties in AI, machine learning, blockchain, and software infrastructure, requiring proficiency in programming languages, "
        "algorithms, and code optimization."
    ),
    
    "Product & Project Management": (
        "Oversees product lifecycle and development strategy, guiding teams to build products that align with market needs and company goals. "
        "Roles manage product roadmaps, prioritize features, and conduct market research to define product vision. "
        "Focus is on collaboration with engineering, design, and marketing teams, balancing technical feasibility with user requirements."
    ),
    
    "Data Science & Machine Learning": (
        "Involves creating models, algorithms, and data analysis techniques to extract insights and make data-driven decisions. "
        "Roles include developing machine learning models, performing statistical analysis, and working with large datasets. "
        "Specializations may involve predictive modeling, NLP, and deep learning, with a focus on experimentation and model optimization."
    ),
    
    "Business Operations & Strategy": (
        "Ensures efficient operations and alignment of resources with company goals. Roles focus on process optimization, performance metrics, "
        "and strategic initiatives to improve business efficiency. Strategy roles may also manage partnerships, business growth, and market expansion, "
        "requiring analytical skills to evaluate business performance and identify new opportunities."
    ),
    
    "Marketing & Growth": (
        "Aims to drive brand awareness and revenue growth through marketing strategies. "
        "Roles develop and execute campaigns, manage digital channels, and analyze engagement metrics to optimize reach. "
        "Includes brand positioning, content strategy, and social media management, focusing on understanding market trends and user behavior."
        "Sales-focused roles may involve lead generation, account management, and closing deals to support business growth."
    ),
    
    "Design & User Experience (UX/UI)": (
        "Dedicated to creating visually appealing and intuitive designs for digital products. Roles involve designing user interfaces, user experiences, "
        "and brand visuals that enhance usability and align with brand identity. Emphasis on user-centered design, prototyping, and aesthetic consistency, "
        "with skills in design tools and principles of layout, color, and typography."
    ),
    
    "Customer Success & Sales": (
        "Focuses on building and maintaining relationships with customers to ensure satisfaction and retention. "
        "Roles include onboarding, client support, and troubleshooting, with a goal to drive product adoption and address customer needs. "
    ),
    
    "Operations & Support": (
        "Supports the efficient functioning of day-to-day business activities. Roles involve administrative tasks, process standardization, and quality management. "
        "Includes internal roles for workflow optimization and ensuring operational consistency. "
        "Skills involve task management, operational metrics, and basic troubleshooting to maintain service reliability and productivity."
    ),
    
    "Engineering Management & Leadership": (
        "Leads and manages technical teams, guiding projects and setting engineering standards. Roles involve decision-making on project priorities, team structuring, "
        "and performance management to align technical work with business goals. Emphasis on team mentorship, resource allocation, and strategic technical direction."
    ),
    
    "Specialized Engineering": (
        "Involves highly specialized technical roles in niche fields requiring domain-specific expertise. Examples include roles in robotics, blockchain, environmental modeling, "
        "and battery management. Focus on advanced, field-specific engineering knowledge and hands-on experience with specialized technologies.Encompasses unique technical roles like Blockchain Architect, Robotics Engineer, Battery Software Engineer, UAV Controls Engineer, and Biogeochemical Modeler, focusing on specific industry and technical expertise."
    ),
    
    "Dev-Ops": (
        "Responsible for optimizing the software deployment lifecycle, from testing and deployment to monitoring and performance tuning. "
        "Roles implement automation, manage infrastructure, and enhance reliability and scalability of systems. Key activities include setting up CI/CD pipelines, "
        "configuring cloud environments, and monitoring for uptime and system resilience."
    )
}

def embed_roles(role_category_description):
    role_category_embeddings = {}
    for key, value in role_category_description.items():
        query_result = embeddings.embed_query(value)
        role_category_embeddings[key] = query_result
    return role_category_embeddings

def upload_pinecone(role_category_embeddings):
    for key, embedding in role_category_embeddings.items():
        index.upsert(
            vectors=[
            {
                "id": key, 
                "values": embedding 
            }
            ]
        )

role_category_embeddings = embed_roles(role_category_description)
upload_pinecone(role_category_embeddings)


