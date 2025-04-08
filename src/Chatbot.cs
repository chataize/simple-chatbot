using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;
using ChatAIze.GenerativeCS.Constants;
using ChatAIze.GenerativeCS.Models;
using ChatAIze.GenerativeCS.Options.OpenAI;
using ChatAIze.SemanticIndex;

namespace ChatAIze.SimpleChatbot;

public sealed class Chatbot
{
    private readonly OpenAIClient _client = new();

    private readonly SemanticDatabase<string> _instructionsDb;

    private readonly SemanticDatabase<string> _knowledgeDb;

    private readonly ChatCompletionOptions _completionOptions = new();

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public Chatbot()
    {
        _instructionsDb = new(_client);
        _knowledgeDb = new(_client);
    }

    public string LanguageModel { get; set; } = ChatCompletionModels.OpenAI.GPT4o;

    public string? InstructionsFilePath { get; set; }

    public string? KnowledgeFilePath { get; set; }

    public int NumberOfRetrievedInstructions { get; set; } = 5;

    public int NumberOfRetrievedKnowledge { get; set; } = 5;

    public async Task<string> CompleteAsync(Chat chat, CancellationToken cancellationToken = default)
    {
        var embedding = await CalculateEmbeddingAsync(chat, cancellationToken);

        var instructions = _instructionsDb.Search(embedding, NumberOfRetrievedInstructions);
        var knowledge = _knowledgeDb.Search(embedding, NumberOfRetrievedKnowledge);

        var options = _completionOptions with
        {
            Model = LanguageModel,
            SystemMessageCallback = () =>
            {
                var prompt = new
                {
                    Instructions = instructions,
                    Knowledge = knowledge,
                };

                var promptJson = JsonSerializer.Serialize(prompt, _jsonOptions);
                return promptJson;
            },
        };

        return await _client.CompleteAsync(chat, options, cancellationToken: cancellationToken);
    }

    public async Task AddInstructionAsync(string instruction, CancellationToken cancellationToken = default)
    {
        await _instructionsDb.AddAsync(instruction, cancellationToken);
    }

    public void RemoveInstruction(string instruction)
    {
        _instructionsDb.Remove(instruction);
    }

    public async Task LoadAsync(CancellationToken cancellationToken = default)
    {
        if (InstructionsFilePath is not null)
        {
            await _instructionsDb.LoadAsync(InstructionsFilePath, cancellationToken);
        }

        if (KnowledgeFilePath is not null)
        {
            await _knowledgeDb.LoadAsync(KnowledgeFilePath, cancellationToken);
        }
    }

    public async Task SaveAsync(CancellationToken cancellationToken = default)
    {
        if (InstructionsFilePath is not null)
        {
            await _instructionsDb.SaveAsync(InstructionsFilePath, cancellationToken);
        }

        if (KnowledgeFilePath is not null)
        {
            await _knowledgeDb.SaveAsync(KnowledgeFilePath, cancellationToken);
        }
    }

    private async Task<float[]> CalculateEmbeddingAsync(Chat chat, CancellationToken cancellationToken = default)
    {
        const int maxMessages = 10;
        const float decayFactor = 0.25f;

        var messages = chat.Messages.Where(m => !string.IsNullOrWhiteSpace(m?.Content)).TakeLast(maxMessages).ToArray();

        if (messages.Length == 0)
        {
            var zeroVector = await _client.GetEmbeddingAsync(string.Empty, cancellationToken: cancellationToken);
            Array.Fill(zeroVector, 0f);

            return zeroVector;
        }

        var tasks = messages.Select(msg => _client.GetEmbeddingAsync(msg!.Content!, cancellationToken: cancellationToken));
        var allEmbeddings = await Task.WhenAll(tasks);
        var weightedSum = new float[allEmbeddings[0].Length];
        var totalWeight = 0.0f;

        for (var i = 0; i < allEmbeddings.Length; i++)
        {
            var distanceFromEnd = messages.Length - 1 - i;
            var weight = MathF.Pow(decayFactor, distanceFromEnd);
            var embedding = allEmbeddings[i];

            for (var j = 0; j < embedding.Length; j++)
            {
                weightedSum[j] += weight * embedding[j];
            }

            totalWeight += weight;
        }

        for (var j = 0; j < weightedSum.Length; j++)
        {
            weightedSum[j] /= totalWeight;
        }

        return weightedSum;
    }
}
