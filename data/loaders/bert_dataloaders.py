from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch



class BertTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]
        text_label=self.dataset[idx]["text_label"]

        # Tokenize single sample (no padding here)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,  # Automatically add [CLS]/[SEP]
            return_tensors="pt"
        )

        # Each sample maintains its own length
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
    




class DynamicPaddingCollator:
    """Dynamic padding collator that pads to the max sequence length per batch."""

    def __init__(self, tokenizer):
        """Store tokenizer to access pad token id."""
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        """Apply dynamic padding to a batch before tensorization."""
        # Extract fields
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

        # Longest sequence length for the batch
        max_len = max(x.size(0) for x in input_ids_list)

        # Pad sequences to max_len
        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - input_ids.size(0)

            # Pad input_ids with pad token id
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )

            # Pad attention mask with zeros
            padded_attention_masks.append(
                torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
            )

        # Stack into tensors
        input_ids = torch.stack(padded_input_ids)
        attention_masks = torch.stack(padded_attention_masks)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }


if __name__ == "__main__":
    """Smoke-test BertTextDataset and DynamicPaddingCollator."""
    import sys
    sys.path.append("/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp")
    
    from datasets import Dataset as HFDataset
    
    print("="*80)
    print("BERT DataLoader Testing")
    print("="*80)
    
    # 1. Build mock data
    print("\n[1] Creating mock dataset...")
    mock_data = {
        "text": [
            "Hello, how are you?",
            "I want to book a flight to Paris.",
            "What's the weather like today?",
            "Can you help me with my order?",
            "This is a very long sentence that should be truncated if it exceeds the maximum length limit set in the tokenizer configuration.",
        ],
        "label": [0, 1, 2, 0, 1],
        "text_label": ["greeting", "flight_booking", "weather", "greeting", "flight_booking"]
    }
    
    hf_dataset = HFDataset.from_dict(mock_data)
    print(f"   - Dataset size: {len(hf_dataset)}")
    print(f"   - Features: {hf_dataset.features}")
    
    # 2. Load tokenizer
    print("\n[2] Loading BERT tokenizer...")
    tokenizer_path = "/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(f"   - Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   - Vocab size: {tokenizer.vocab_size}")
    print(f"   - PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"   - CLS token: '{tokenizer.cls_token}' (id={tokenizer.cls_token_id})")
    print(f"   - SEP token: '{tokenizer.sep_token}' (id={tokenizer.sep_token_id})")
    
    # 3. Create Dataset
    print("\n[3] Creating BertTextDataset...")
    max_length = 32
    dataset = BertTextDataset(hf_dataset, tokenizer, max_length=max_length)
    print(f"   - Dataset length: {len(dataset)}")
    print(f"   - Max length: {max_length}")
    
    # 4. Inspect single samples
    print("\n[4] Testing single sample retrieval...")
    sample_0 = dataset[0]
    sample_4 = dataset[4]
    
    print(f"\n   Sample 0 (short text):")
    print(f"   - Text: '{mock_data['text'][0]}'")
    print(f"   - input_ids shape: {sample_0['input_ids'].shape}")
    print(f"   - input_ids: {sample_0['input_ids']}")
    print(f"   - attention_mask: {sample_0['attention_mask']}")
    print(f"   - label: {sample_0['labels'].item()} ({sample_0['text_label']})")
    print(f"   - Decoded: '{tokenizer.decode(sample_0['input_ids'])}'")
    
    print(f"\n   Sample 4 (long text, should be truncated):")
    print(f"   - Text: '{mock_data['text'][4][:50]}...'")
    print(f"   - input_ids shape: {sample_4['input_ids'].shape}")
    print(f"   - input_ids length: {len(sample_4['input_ids'])}")
    print(f"   - label: {sample_4['labels'].item()} ({sample_4['text_label']})")
    
    # 5. Create DataLoader with DynamicPaddingCollator
    print("\n[5] Creating DataLoader with DynamicPaddingCollator...")
    collator = DynamicPaddingCollator(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=3, 
        shuffle=False, 
        collate_fn=collator
    )
    print(f"   - Batch size: 3")
    print(f"   - Number of batches: {len(dataloader)}")
    
    # 6. Inspect first batch
    print("\n[6] Testing first batch...")
    first_batch = next(iter(dataloader))
    
    print(f"   Batch keys: {first_batch.keys()}")
    print(f"   - input_ids shape: {first_batch['input_ids'].shape}")
    print(f"   - attention_mask shape: {first_batch['attention_mask'].shape}")
    print(f"   - labels shape: {first_batch['labels'].shape}")
    print(f"   - labels: {first_batch['labels']}")
    
    # Check padding
    print(f"\n   Checking dynamic padding:")
    for i in range(first_batch['input_ids'].size(0)):
        seq = first_batch['input_ids'][i]
        mask = first_batch['attention_mask'][i]
        actual_len = mask.sum().item()
        num_pads = (seq == tokenizer.pad_token_id).sum().item()
        print(f"   - Sample {i}: actual_len={actual_len}, num_pads={num_pads}, total_len={len(seq)}")
        print(f"     Text: '{tokenizer.decode(seq[mask.bool()], skip_special_tokens=False)}'")
    
    # 7. Iterate over batches
    print("\n[7] Testing all batches...")
    all_labels = []
    for batch_idx, batch in enumerate(dataloader):
        print(f"   Batch {batch_idx}:")
        print(f"   - input_ids shape: {batch['input_ids'].shape}")
        print(f"   - Max sequence length in batch: {batch['input_ids'].size(1)}")
        print(f"   - Labels: {batch['labels'].tolist()}")
        all_labels.extend(batch['labels'].tolist())
    
    print(f"\n   All labels collected: {all_labels}")
    print(f"   Expected labels: {mock_data['label']}")
    print(f"   Labels match: {all_labels == mock_data['label']}")
    
    # 8. Validate padding correctness
    print("\n[8] Validating padding correctness...")
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Validate attention mask vs pad token
        for i in range(input_ids.size(0)):
            seq = input_ids[i]
            mask = attention_mask[i]
            
            # Find first pad token index
            pad_positions = (seq == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            
            if len(pad_positions) > 0:
                first_pad_pos = pad_positions[0].item()
                # Ensure pad positions have mask 0
                assert mask[first_pad_pos] == 0, f"Batch {batch_idx}, Sample {i}: pad position has attention_mask=1"
                # Ensure every pad position mask is 0
                assert (mask[pad_positions] == 0).all(), f"Batch {batch_idx}, Sample {i}: some pads have attention_mask=1"
            
            # Ensure mask==1 positions are non-pad
            non_pad_positions = (mask == 1).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                assert (seq[non_pad_positions] != tokenizer.pad_token_id).all(), \
                    f"Batch {batch_idx}, Sample {i}: non-pad positions contain pad tokens"
    
    print("   ✓ All padding validation passed!")
    
    # 9. Test different batch size
    print("\n[9] Testing different batch sizes...")
    for bs in [1, 2, 5]:
        dl = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collator)
        first_batch = next(iter(dl))
        print(f"   Batch size={bs}: input_ids shape = {first_batch['input_ids'].shape}")
    
    print("\n" + "="*80)
    print("✓ All tests passed successfully!")
    print("="*80)


