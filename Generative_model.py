from transformers import pipeline

# Load the T5-base summarization pipeline
summarizer = pipeline("summarization", model="t5-base")

# Example network event with 15 key features
sample_row = {
    "Packet Length Variance": 2500,
    "Average Packet Size": 60,
    "Bwd Packet Length Std": 1200,
    "Avg Bwd Segment Size": 500,
    "Destination Port": 80,
    "Packet Length Std": 1000,
    "Packet Length Mean": 55,
    "Total Length of Fwd Packets": 30,
    "Bwd Packet Length Mean": 0,
    "Subflow Fwd Bytes": 30,
    "Subflow Bwd Bytes": 0,
    "Max Packet Length": 6,
    "Init_Win_bytes_forward": 256,
    "Bwd Packet Length Max": 0,
    "Fwd Packet Length Mean": 6,
    "Label": "Attack"
}

# Construct input text for better summarization performance
input_text = f"""
A network flow was detected with the following characteristics:
- Destination Port: {sample_row['Destination Port']}
- Packet Length Variance: {sample_row['Packet Length Variance']} bytes squared
- Average Packet Size: {sample_row['Average Packet Size']} bytes
- Backward Packet Length Std: {sample_row['Bwd Packet Length Std']} bytes
- Average Backward Segment Size: {sample_row['Avg Bwd Segment Size']} bytes
- Packet Length Std: {sample_row['Packet Length Std']} bytes
- Packet Length Mean: {sample_row['Packet Length Mean']} bytes
- Total Length of Forward Packets: {sample_row['Total Length of Fwd Packets']} bytes
- Backward Packet Length Mean: {sample_row['Bwd Packet Length Mean']} bytes
- Subflow Forward Bytes: {sample_row['Subflow Fwd Bytes']} bytes
- Subflow Backward Bytes: {sample_row['Subflow Bwd Bytes']} bytes
- Max Packet Length: {sample_row['Max Packet Length']} bytes
- Initial Window Bytes Forward: {sample_row['Init_Win_bytes_forward']} bytes
- Backward Packet Length Max: {sample_row['Bwd Packet Length Max']} bytes
- Forward Packet Length Mean: {sample_row['Fwd Packet Length Mean']} bytes

This flow is labeled as: {sample_row['Label']}
"""

# T5 expects "summarize: " prefix for best results
formatted_input = f"summarize: {input_text}"

# Generate summary with T5-base
summary = summarizer(formatted_input, max_length=200, min_length=20, do_sample=False)

# Print the generated summary
print("\nGenerated Summary:")
print(summary[0]['summary_text'])
