from __future__ import annotations
import sys, os, json
from flask import Flask, request
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.planning.plan import Plan
from azure.identity import DefaultAzureCredential, AzureCliCredential

# add parent directory to path
sys.path.insert(0,  str(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))

import utils
openai_config:utils.OpenAIConfig=None

IS_CHAT_COMPLETION = None

import importlib
importlib.reload(utils)

# set to true for chat completion API, false for text completion
IS_CHAT_COMPLETION=True
credential = DefaultAzureCredential(additionally_allowed_tenants = ['*'])

def init() -> tuple [sk.Kernel, BasicPlanner]:
    utils.load_secrets(credential)
    load_env_vars()

    kernel = create_kernel(debug=False)
    planner = BasicPlanner()
    return kernel, planner

def load_env_vars():
    global openai_config
    openai_config = utils.OpenAIConfig.from_env()
    global IS_CHAT_COMPLETION
    IS_CHAT_COMPLETION = bool(os.environ.get('IS_CHAT_COMPLETION'))

def import_skills(kernel: sk.Kernel, skills_folder: str):
    print(f"Importing skills from {skills_folder}")
    for skill_name in os.listdir(skills_folder):
        skill_full_path = os.path.join(skills_folder, skill_name)
        print(f"== Importing skill {skill_name}: {skill_full_path}")
        kernel.import_semantic_skill_from_directory(skills_folder, skill_name)

def create_kernel(debug: bool = False) -> sk.Kernel: 
    logger = sk.NullLogger()
    if(debug):
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(handler)

    kernel = sk.Kernel()
    if openai_config.OPENAI_API_TYPE == 'azure':
        # if using chat service from Azure OpenAI API, use AzureChatCompletion
        kernel.add_text_completion_service(
            "completion",     
            sk_oai.AzureChatCompletion(
                deployment_name=openai_config.AZURE_OPENAI_API_DEPLOYMENT_NAME,
                api_key=openai_config.OPENAI_API_KEY,
                endpoint=openai_config.AZURE_OPENAI_API_ENDPOINT
            ) if IS_CHAT_COMPLETION else
            sk_oai.AzureTextCompletion(
                deployment_name=openai_config.AZURE_OPENAI_API_DEPLOYMENT_NAME, 
                api_key=openai_config.OPENAI_API_KEY, 
                endpoint=openai_config.AZURE_OPENAI_API_ENDPOINT
            ),
        )
    else:
        print('using openai', openai_config.OPENAI_MODEL_ID, openai_config.OPENAI_ORG_ID)
        kernel.add_text_completion_service(
            "completion",
            sk_oai.OpenAIChatCompletion(
                openai_config.OPENAI_MODEL_ID, openai_config.OPENAI_API_KEY, openai_config.OPENAI_ORG_ID
            ) if IS_CHAT_COMPLETION else
            sk_oai.OpenAITextCompletion(
                openai_config.OPENAI_MODEL_ID, openai_config.OPENAI_API_KEY, openai_config.OPENAI_ORG_ID
            ),
        )

    # import skills from skills folder
    import_skills(kernel, os.path.join(os.path.dirname(os.path.realpath(__file__)), "skills"))

    return kernel

kernel, planner = init()

async def invoke_skill(skillName, functionName, context): 
    skillFunction = kernel.func(skillName, functionName)
    return await skillFunction.invoke_async(context=context)

# class for plan deserializing
class GeneratedPlan:
    def __init__(self, result: str):
        self.result = result

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "ok"

@app.route('/health', methods=['GET'])
def health():
    return "healthy"

@app.route('/skills/<skillName>/invoke/<functionName>', methods=['POST'])
async def invoke(skillName, functionName):
    return await invoke_skill(skillName, functionName, request.get_json())

@app.route('/planner/createplan', methods=['POST'])
async def createplan():
    body = request.get_json()
    goal = body["value"]
    plan = await planner.create_plan_async(goal, kernel)
    print(plan.generated_plan.result)
    return plan.generated_plan.result

@app.route('/planner/executeplan', methods=['POST'])
async def executeplan():
    body = request.get_json()
    print(body)
    gp = GeneratedPlan(result = json.dumps(body))
    p = Plan(goal=None, prompt=None, plan=gp)

    result = await planner.execute_plan_async(p, kernel)
    print(result)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
