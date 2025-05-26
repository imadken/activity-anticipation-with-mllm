from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

from neo4j import GraphDatabase, basic_auth

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv('DEEP_SEEK_API_KEY'),
)


class KG:
    def __init__(self, uri="neo4j+ssc://3ca39d1b.databases.neo4j.io", user="neo4j", password=getenv('NEO4J_PASSWORD')):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Connected to Neo4j Aura")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Neo4j: {e}")

    def close(self):
        self.driver.close()

    def add_object_with_affordances(self, object_name: str, affordances: list[str]):
        query = """
        MERGE (o:Object {name: $object})
        FOREACH (a IN $affordances |
            MERGE (aff:Affordance {name: a})
            MERGE (o)-[:CAN_BE_USED_TO]->(aff)
        )
        """
        with self.driver.session(database="neo4j") as session:
            session.run(query, object=object_name, affordances=affordances)

    def get_affordances_for_object(self, object_name: str) -> list[str]:
        query = """
        MATCH (:Object {name: $object})-[:CAN_BE_USED_TO]->(a:Affordance)
        RETURN a.name AS affordance
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, object=object_name)
            return [record["affordance"] for record in result]

    def get_objects_for_affordance(self, affordance: str) -> list[str]:
        query = """
        MATCH (o:Object)-[:CAN_BE_USED_TO]->(:Affordance {name: $affordance})
        RETURN o.name AS object
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, affordance=affordance)
            return [record["object"] for record in result]

    def get_all_data(self) -> list[tuple[str, str]]:
        query = """
        MATCH (o:Object)-[:CAN_BE_USED_TO]->(a:Affordance)
        RETURN o.name AS object, a.name AS affordance
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query)
            return [(record["object"], record["affordance"]) for record in result]

    def object_exists(self, object_name: str) -> bool:
        query = """
        MATCH (o:Object {name: $object})
        RETURN COUNT(o) > 0 AS exists
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, object=object_name)
            return result.single()["exists"]


def get_response_openai(prompt, model="deepseek/deepseek-prover-v2:free", MLLM=False):
    if not MLLM:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = (
    "You are an expert in building knwowledge graphs.\n"
    "Given an object, you will return the affordances that can be done with it.\n"
    "Return the answer in the following format separated by commas:\n"
    "affordance1, affordance2, ...\n"
    "Return only the answer, do not include any other text or explanation and keep the answers simple (only one word or verb if possible).\n"
)

if __name__ == "__main__":
    kg = KG()
    objects = ["banana", "spoon", "knife", "fork", "cup", "bottle", "pen", "paper", "book", "phone"]

    for obj in objects:
        if not kg.object_exists(obj):
            print(f"Adding {obj} to the knowledge graph.")
            affordances = get_response_openai(f"{SYSTEM_PROMPT} what are the affordances of a {obj}?").replace('\n', '').replace('"', '').replace("```", '').replace("python", '').split(",")

            print(f"Affordances for {obj}: {affordances}")
            
            affordances = [a.strip() for a in affordances if a.strip()]


            kg.add_object_with_affordances(obj, affordances)
        else:
            print(f"{obj} already exists in the knowledge graph.")

    kg.close()
