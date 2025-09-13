# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
from MovieAgentV2 import MovieAgentV2

if __name__ == '__main__':
    try:
        client = MovieAgentV2()
        client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Unexpected error: {e}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
