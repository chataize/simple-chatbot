using ChatAIze.GenerativeCS.Models;
using ChatAIze.SimpleChatbot;

var chatbot = new Chatbot();

await chatbot.AddInstructionAsync("If the user asks about animals, always tell them about monkeys.");
await chatbot.AddInstructionAsync("If the user asks about food, always tell them about eggs.");
await chatbot.AddInstructionAsync("If the user asks about the weather, always tell them it's stormy.");
await chatbot.AddInstructionAsync("If the user asks about the time, always tell them it's 3 PM.");
await chatbot.AddInstructionAsync("If the user asks about the date, always tell them it's 13st January 2023.");

Console.WriteLine("Ready!");

var chat = new Chat();

while (true)
{
    var message = Console.ReadLine()!;
    chat.FromUser(message);

    var response = await chatbot.CompleteAsync(chat);
    Console.WriteLine(response);
}
