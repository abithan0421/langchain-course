import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")

    information = """
    BTS (Korean: 방탄소년단; RR: Bangtan sonyeondan; lit. 'Bulletproof Boy Scouts'), also known as the Bangtan Boys, is a South Korean boy band formed in 2010. The band consists of Jin, Suga, J-Hope, RM, Jimin, V, and Jung Kook, who co-write or co-produce much of their material. Originally a hip hop group, they expanded their musical style to incorporate a wide range of genres, while their lyrics have focused on subjects including mental health, the troubles of school-age youth and coming of age, loss, the journey towards self-love, individualism and the consequences of fame and recognition. Their discography and adjacent work has also referenced literature, philosophy and psychology, and includes an alternate universe storyline.

    BTS debuted in 2013 under Big Hit Entertainment with the single album 2 Cool 4 Skool. BTS released their first Korean and Japanese-language studio albums, Dark & Wild and Wake Up respectively, in 2014. The group's second Korean studio album, Wings (2016), was their first to sell one million copies in South Korea. By 2017, BTS had crossed into the global music market and led the Korean Wave into the United States, becoming the first Korean ensemble to receive a Gold certification from the Recording Industry Association of America (RIAA) for their single "Mic Drop", as well as the first act from South Korea to top the Billboard 200 with their studio album Love Yourself: Tear (2018). In 2020, BTS became the fastest group since the Beatles to chart four US number-one albums in less than two years, with Love Yourself: Answer (2018) becoming the first Korean album certified Platinum by the RIAA; in the same year, they also became the first all-South Korean act to reach number one on both the Billboard Hot 100 and Billboard Global 200 with their Grammy-nominated single "Dynamite". Follow-up releases "Savage Love", "Life Goes On", "Butter", "Permission to Dance", and "My Universe" made them the fastest act to earn six US number-one singles since the Beatles in 1966.
    """

    summary_template = """
    Given the information {information} about a person, I want you to provide 
    1. A short summary
    2. Two intersting facts about the person
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    # llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm = ChatOllama(temperature = 0, model = "gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
