from dotenv import load_dotenv

from graph.graph import app

load_dotenv()


def main():
    print("Hello from langchain-training!")
    output = app.invoke(input={"question": "What is agent memory?"})
    print(output)


if __name__ == "__main__":
    main()
