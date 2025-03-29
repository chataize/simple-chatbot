using ChatAIze.GenerativeCS.Clients;
using ChatAIze.GenerativeCS.Constants;
using ChatAIze.GenerativeCS.Models;
using ChatAIze.GenerativeCS.Options.OpenAI;

namespace ChatAIze.SimpleChatbot;

public sealed class Chatbot
{
    private readonly OpenAIClient _client = new();

    private readonly ChatCompletionOptions _options = new();

    public string LanguageModel { get; set; } = ChatCompletionModels.OpenAI.GPT4o;

    public async Task<string> CompleteAsync(Chat chat)
    {
        var options = _options with
        {
            Model = LanguageModel
        };

        return await _client.CompleteAsync(chat, options);
    }
}
