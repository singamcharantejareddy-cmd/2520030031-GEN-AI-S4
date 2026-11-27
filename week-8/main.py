
import torch
import torch.nn as nn
import math

# 1. Scaled Dot-Product Attention

def scaled_dot_product_attention(Q, K, V, causal=False):

    # Calculate Query and Key transpose
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale the scores
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)

    # Apply causal mask if required
    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float('-inf'))

    # Convert scores into attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Multiply attention weights with Value
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# 2. Create Query, Key and Value

Q = torch.randn(1, 3, 4)
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)


# 3. Apply Scaled Dot-Product Attention

output, weights = scaled_dot_product_attention(Q, K, V)

print("Scaled Dot-Product Attention")
print("--------------------------------")

print("Query Shape:", Q.shape)
print("Key Shape:", K.shape)
print("Value Shape:", V.shape)

print("\nAttention Weights:")
print(weights)

print("\nAttention Output:")
print(output)

print("\nOutput Shape:", output.shape)


# 4. Multi-Head Attention

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Query, Key and Value
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, causal=False):

        batch_size, seq_len, embed_dim = x.shape

        # Generate Query, Key and Value
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split into multiple attention heads
        Q = Q.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Apply Scaled Dot-Product Attention
        attention_output, attention_weights = (
            scaled_dot_product_attention(Q, K, V, causal)
        )

        # Combine all attention heads
        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # Final linear projection
        output = self.W_O(attention_output)

        return output, attention_weights


# 5. Test Multi-Head Attention

print("\n\nMulti-Head Attention")
print("--------------------------------")

embed_dim = 8
num_heads = 2

# Create random input embeddings
x = torch.randn(1, 3, embed_dim)

# Initialize Multi-Head Attention
mha = MultiHeadAttention(embed_dim, num_heads)

# Perform attention
mha_output, mha_weights = mha(x)

print("Input Shape:", x.shape)
print("Number of Heads:", num_heads)

print("Attention Weights Shape:", mha_weights.shape)

print("\nMulti-Head Attention Output:")
print(mha_output)

print("\nOutput Shape:", mha_output.shape)
