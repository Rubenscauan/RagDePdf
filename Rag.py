import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools  import tool
from crewai import Crew
from crewai import Task
from crewai import Agent


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)


rag_tool = PDFSearchTool(pdf='lei.pdf',
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3-8b-8192",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)



web_search_tool = TavilySearchResults(k=5)


@tool
def router_tool(question):
  """Router Function"""
  if 'lei' in question:
    return 'vectorstore'
  else:
    return 'web_search'
     



Agente_Enrutador = Agent(
    role='Enrutador',
    goal='Redirecionar a pergunta do usuário para uma vectorstore ou busca na web',
    backstory=(
        "Você é especialista em redirecionar perguntas de usuários para uma vectorstore ou busca na web."
        "Use a vectorstore para perguntas relacionadas a Recuperação e Geração Aumentada (Retrieval-Augmented Generation)."
        "Não precisa ser rigoroso com as palavras-chave nas perguntas relacionadas a esses tópicos. Caso contrário, use a busca na web."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Agente_Recuperador = Agent(
    role="Recuperador",
    goal="Usar as informações recuperadas da vectorstore para responder à pergunta",
    backstory=(
        "Você é um assistente para tarefas de perguntas e respostas."
        "Use as informações presentes no contexto recuperado para responder à pergunta."
        "Você deve fornecer uma resposta clara e concisa."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Agente_Avaliador = Agent(
    role='Avaliador de Resposta',
    goal='Filtrar recuperações errôneas',
    backstory=(
        "Você é um avaliador que analisa a relevância de um documento recuperado em relação à pergunta do usuário."
        "Se o documento contém palavras-chave relacionadas à pergunta, avalie como relevante."
        "Não precisa ser um teste rigoroso. Você deve garantir que a resposta seja relevante para a pergunta."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Avaliador_Alucinação = Agent(
    role="Avaliador de Alucinação",
    goal="Filtrar alucinações",
    backstory=(
        "Você é um avaliador de alucinação que verifica se a resposta é fundamentada em / suportada por um conjunto de fatos."
        "Certifique-se de revisar minuciosamente a resposta e verificar se está alinhada com a pergunta feita."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Avaliador_Resposta = Agent(
    role="Avaliador de Resposta",
    goal="Filtrar alucinações na resposta.",
    backstory=(
        "Você é um avaliador que verifica se uma resposta é útil para resolver uma pergunta."
        "Certifique-se de revisar meticulosamente a resposta e verificar se ela faz sentido para a pergunta feita."
        "Se a resposta for relevante, gere uma resposta clara e concisa."
        "Se a resposta gerada não for relevante, realize uma busca na web usando 'web_search_tool'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

tarefa_enrutador = Task(
    description=("Analise as palavras-chave na pergunta {question}."
    "Com base nas palavras-chave, decida se ela é elegível para uma busca em vectorstore ou na web."
    "Retorne uma única palavra 'vectorstore' se for elegível para vectorstore."
    "Retorne uma única palavra 'websearch' se for elegível para busca na web."
    "Não forneça nenhuma introdução ou explicação."),
    expected_output=("Dê uma escolha binária 'websearch' ou 'vectorstore' com base na pergunta."
    "Não forneça nenhuma introdução ou explicação."),
    agent=Agente_Enrutador,
    tools=[router_tool],
)

tarefa_recuperador = Task(
    description=("Com base na resposta da tarefa de enrutamento, extraia informações para a pergunta {question} com a ajuda da ferramenta respectiva."
    "Use a ferramenta web_search_tool para recuperar informações da web caso a saída da tarefa de enrutamento seja 'websearch'."
    "Use a ferramenta rag_tool para recuperar informações da vectorstore caso a saída da tarefa de enrutamento seja 'vectorstore'."),
    expected_output=("Você deve analisar a saída da 'tarefa_enrutador'."
    "Se a resposta for 'websearch', use a ferramenta web_search_tool para recuperar informações da web."
    "Se a resposta for 'vectorstore', use a ferramenta rag_tool para recuperar informações da vectorstore."
    "Retorne um texto claro e conciso como resposta."),
    agent=Agente_Recuperador,
    context=[tarefa_enrutador],
)

tarefa_avaliador = Task(
    description=("Com base na resposta da tarefa de recuperação para a pergunta {question}, avalie se o conteúdo recuperado é relevante para a pergunta."),
    expected_output=("Nota binária 'sim' ou 'não' indicando se o documento é relevante para a pergunta."
    "Você deve responder 'sim' se a resposta da 'tarefa_recuperador' estiver alinhada com a pergunta feita."
    "Você deve responder 'não' se a resposta da 'tarefa_recuperador' não estiver alinhada com a pergunta feita."
    "Não forneça nenhuma introdução ou explicação, exceto por 'sim' ou 'não'."),
    agent=Agente_Avaliador,
    context=[tarefa_recuperador],
)

tarefa_alucinação = Task(
    description=("Com base na resposta da tarefa de avaliação para a pergunta {question}, avalie se a resposta é fundamentada em / suportada por um conjunto de fatos."),
    expected_output=("Nota binária 'sim' ou 'não' indicando se a resposta está alinhada com a pergunta feita."
    "Responda 'sim' se a resposta for útil e contiver fatos sobre a pergunta feita."
    "Responda 'não' se a resposta não for útil e não contiver fatos sobre a pergunta feita."
    "Não forneça nenhuma introdução ou explicação, exceto por 'sim' ou 'não'."),
    agent=Avaliador_Alucinação,
    context=[tarefa_avaliador],
)

tarefa_resposta = Task(
    description=("Com base na resposta da tarefa de alucinação para a pergunta {question}, avalie se a resposta é útil para resolver a pergunta."
    "Se a resposta for 'sim', retorne uma resposta clara e concisa."
    "Se a resposta for 'não', realize uma 'websearch' e retorne a resposta."),
    expected_output=("Retorne uma resposta clara e concisa se a resposta da 'tarefa_alucinação' for 'sim'."
    "Realize uma busca na web usando 'web_search_tool' e retorne uma resposta clara e concisa somente se a resposta da 'tarefa_alucinação' for 'não'."
    "Caso contrário, responda com 'Desculpe! Não foi possível encontrar uma resposta válida'."
    ),
    context=[tarefa_alucinação],
    agent=Avaliador_Resposta
)

equipe_rag = Crew(
    agents=[Agente_Enrutador, Agente_Recuperador, Agente_Avaliador, Avaliador_Alucinação, Avaliador_Resposta],
    tasks=[tarefa_enrutador, tarefa_recuperador, tarefa_avaliador, tarefa_alucinação, tarefa_resposta],
    verbose=True,
)


inputs = {"question": "Qual seria a punição por infrigir a lei tratada nesse pdf (responda a pergunta)?"}

resultado = equipe_rag.kickoff(inputs=inputs)
print(resultado)
