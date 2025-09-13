import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_user_past_reviews(user_id):
    """Simulate fetching past reviews of a user from a database or API."""
    simulated_reviews = {
        "user1": [
            {"movie": "Inception", "id": 101, "review": "Amazing plot and visuals!"},
            {"movie": "The Matrix", "id": 102, "review": "A mind-bending experience."},
            {"movie": "Jurassic Park", "id": 103, "review": "One time watch movie. Wont watch it again"},
            {"movie": "Godfather", "id": 104, "review": "Boring movie"}
        ],
        "user2": [
            {"movie": "Titanic", "review": "A heartbreaking love story."},
            {"movie": "Avatar", "review": "Stunning world-building and effects."}
        ]
    }
    return simulated_reviews[user_id]


def get_genres(movie_ids):
    """Simulate fetching genres for a list of movie IDs."""
    simulated_genres = {
        101: ["Sci-Fi", "Thriller"],
        102: ["Sci-Fi", "Action"],
        103: ["Adventure"],
        104: ["Crime", "Drama"],
        201: ["Romance", "Drama"],
        202: ["Sci-Fi", "Adventure"]
    }

    resultant_movie_genres_set = set()
    for movie_id in movie_ids:
        if movie_id in simulated_genres:
            resultant_movie_genres_set.update(simulated_genres[movie_id])

    return list(resultant_movie_genres_set)


def get_movies(genres, watched_movie_ids):
    """Simulate fetching movies based on genres and excluding watched movies."""
    simulated_genres_movies = {
        "Sci-Fi": [
            {"id": 201, "title": "Interstellar"},
            {"id": 202, "title": "The Martian"}
        ],
        "Adventure": [
            {"id": 203, "title": "Avatar"},
            {"id": 204, "title": "Guardians of the Galaxy"}
        ],
        "Thriller": [
            {"id": 205, "title": "Tenet"},
            {"id": 101, "title": "Inception"},
            {"id": 102, "title": "The Matrix"},
            {"id": 208, "title": "Edge of Tomorrow"}
        ],
        "Crime": [
            {"id": 209, "title": "Pulp Fiction"},
            {"id": 210, "title": "The Dark Knight"}
        ]
    }
    recommended_movies = []

    for genre in genres:
        if genre in simulated_genres_movies:
            for movie in simulated_genres_movies[genre]:
                if movie['id'] not in watched_movie_ids:
                    recommended_movies.append(movie['title'])

    return recommended_movies


class MovieAgentV2:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_user_past_reviews",
                "description": "Get user past reviews for a movie",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user ID to fetch reviews for past watched movies"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_genres",
                "description": "Get genres for the list of movies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_ids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of movie IDs to fetch genres for"
                        }
                    },
                    "required": ["movie_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_movies",
                "description": "Get movies based on genre ids which user likes and has not watched it yet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "genres": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of genres to fetch movies for"
                        },
                        "watched_movie_ids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of passed movie ids user has watched"
                        }
                    },
                    "required": ["genres", "watched_movie_ids"]
                }
            }
        }
    ]

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o"
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def chat_completion_with_tools(self,
                                   user_message: str,
                                   system_message=None,
                                   max_tokens: int = 200,
                                   temperature: float = 0.7) -> str:
        """
        Make a chat completion request with tool calling capabilities.
        """
        # Build messages list with full conversation history
        messages = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add all previous conversation messages
        messages.extend(self.conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(response)
            response_message = response.choices[0].message

            while response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_call.function.name == "get_user_past_reviews":
                    user_id = tool_args.get("user_id", "Unknown")
                    tool_result = get_user_past_reviews(user_id)

                elif tool_call.function.name == "get_genres":
                    movie_ids = tool_args.get("movie_ids", "Unknown")
                    tool_result = get_genres(movie_ids)

                elif tool_call.function.name == "get_movies":
                    genres = tool_args.get("genres", "Unknown")
                    watched_movie_ids = tool_args.get("watched_movie_ids", "Unknown")
                    tool_result = get_movies(genres, watched_movie_ids)

                else:
                    break

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result)
                })
                self.conversation_history.append(
                    {"role": "tool", "name": tool_name, "content": json.dumps(tool_result)})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.TOOLS,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                print(response)
                response_message = response.choices[0].message

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            ai_response = final_response.choices[0].message.content.strip()

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})

            return ai_response

        except Exception as e:
            return f"Error making API call: {str(e)}"

    def start_conversation(self):
        """
        Start a continuous conversation loop with memory and tool calling.
        """
        print("🤖 Chatbot with Memory and Tools is ready!")
        print("=" * 60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear conversation history")
        print("=" * 60)

        system_message = """You are a helpful AI assistant with access to tools. 
            When user asks you about the watching movie, you respond by asking the user id of that user.
            When user tells the user id, you make simultaneous sequential tool calls before recommending the unwatched movies. Here are the list of tools presnet for recommending movies:
            get_user_reviews tool which take user id as input, get_genres tool which takes movie ids as input, get_movies tool using which takes genres and watched movie ids as input
            Make sure that recommendations are based on the positive reviews and genres fetched using the tools.
            Also make sure that user has not watched that movie before.
            Keep responses concise and engaging. You can reference previous parts of our conversation."""

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 AI: Goodbye! Thanks for chatting!")
                    break

                # Check for special commands
                if user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue

                if user_input.lower() == 'clear':
                    self.clear_conversation_history()
                    continue

                if not user_input:
                    print("Please enter a message.")
                    continue

                # Send to LLM with tool calling capabilities
                print("🔄 Processing...")
                response = self.chat_completion_with_tools(
                    user_message=user_input,
                    system_message=system_message
                )

                # Display response
                print(f"🤖 AI: {response}")

            except KeyboardInterrupt:
                print("\n\n🤖 AI: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_conversation_history(self):
        """Display the current conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return

        print("\n📝 Conversation History:")
        print("-" * 40)
        for i, message in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if message["role"] == "user" else "🤖"
            content = message.get("content", "")

            if message["role"] == "tool":
                role_emoji = "🔧"
                content = f"[Tool Result: {content}]"
            elif message["role"] == "assistant":
                # Check if this is a tool call message
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]
                    tool_info = []
                    for tool_call in tool_calls:
                        tool_info.append(
                            f"🔧 Called {tool_call.function.name} with args: {tool_call.function.arguments}")

                    if content:
                        content = f"{content} {' '.join(tool_info)}"
                    else:
                        content = " ".join(tool_info)

            print(f"{i}. {role_emoji} {message['role'].title()}: {content}")
        print("-" * 40)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        print("🧹 Conversation history cleared!")
