namespace ChatAIze.SimpleChatbot;

public sealed class ChatbotBuilder
{
    private readonly Chatbot _chatbot = new();

    public ChatbotBuilder WithInstructionsFile(string filePath)
    {
        _chatbot.InstructionsFilePath = filePath;
        return this;
    }

    public ChatbotBuilder WithKnowledgeFile(string filePath)
    {
        _chatbot.KnowledgeFilePath = filePath;
        return this;
    }

    public Chatbot Build()
    {
        return _chatbot;
    }
}
