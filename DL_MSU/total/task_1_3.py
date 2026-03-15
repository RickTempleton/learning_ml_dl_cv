import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import argparse
import sys
import random
import os

# Модель для заданий 1 и 2
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, seq_len=20):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        # Правильная булевая маска
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x_emb = self.embed(x) * (self.d_model ** 0.5)
        pos_emb = self.pos_embed(positions)
        x = x_emb + pos_emb

        out = self.decoder(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)
        return self.fc_out(out)

# Модель для задания 3 (с регуляризацией)
class RegularizedCharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, seq_len=20, dropout_rate=0.1, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if tie_weights:
            self.fc_out.weight = self.embed.weight

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.embed(x)
        pos_emb = self.pos_embed(positions)
        x = x_emb + pos_emb
        x = self.dropout(x)
        
        # Правильная булевая маска
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        out = self.decoder(tgt=x, memory=x, tgt_mask=mask)
        return self.fc_out(out)

# Модель для арифметических последовательностей (задание 2)
class ArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, seq_len=10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x_emb = self.embed(x) * (self.d_model ** 0.5)
        pos_emb = self.pos_embed(positions)
        x = x_emb + pos_emb

        out = self.decoder(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)
        return self.fc_out(out)

def generate_word_advanced(model, char2idx, idx2char, start_chars="", max_length=15, temperature=1.0, top_k=5, top_p=0.9):
    """Продвинутая генерация слов с top-k и top-p sampling"""
    device = next(model.parameters()).device
    model.eval()
    
    if start_chars:
        sequence = [char2idx['<sos>']] + [char2idx[char] for char in start_chars if char in char2idx]
    else:
        sequence = [char2idx['<sos>']]
    
    eos_token = char2idx['<eos>']
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(sequence, device=device).unsqueeze(0)
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / max(temperature, 0.1)
            
            # Top-k фильтрация
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Top-p (nucleus) фильтрация
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Исключаем специальные токены
            allowed_indices = [idx for idx in range(len(probs)) 
                             if idx not in [char2idx['<sos>'], char2idx['<eos>'], char2idx['<pad>']]
                             and idx < len(probs)]
            
            if not allowed_indices:
                break
                
            allowed_probs = probs[allowed_indices]
            
            if allowed_probs.sum() > 1e-8:
                allowed_probs = allowed_probs / allowed_probs.sum()
                
                # Добавляем небольшой шум для разнообразия
                if len(allowed_probs) > 1 and temperature > 0.5:
                    noise = torch.randn_like(allowed_probs.float()) * 0.01
                    allowed_probs = torch.softmax(torch.log(allowed_probs) + noise, dim=-1)
                
                next_token_idx = torch.multinomial(allowed_probs, 1).item()
                next_token = allowed_indices[next_token_idx]
            else:
                next_token = random.choice(allowed_indices)
            
            sequence.append(next_token)
            
            if next_token == eos_token or len(sequence) >= max_length + 1:
                break
    
    generated_word = ''.join([idx2char[idx] for idx in sequence 
                            if idx2char[idx] not in ['<sos>', '<eos>', '<pad>']])
    
    return generated_word if generated_word else ""

def is_arithmetic_progression(sequence):
    """Проверяет, является ли последовательность арифметической прогрессией"""
    if len(sequence) < 2:
        return False, 0
    
    differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
    first_diff = differences[0]
    
    # Проверяем, все ли разности одинаковы
    is_arithmetic = all(diff == first_diff for diff in differences)
    
    return is_arithmetic, first_diff

def analyze_sequence_errors(sequence):
    """Анализирует последовательность и находит места ошибок"""
    if len(sequence) < 2:
        return [], 0
    
    errors = []
    
    # Находим наиболее вероятную разность из начала последовательности
    if len(sequence) >= 3:
        diffs = [sequence[i+1] - sequence[i] for i in range(min(3, len(sequence)-1))]
        most_common_diff = max(set(diffs), key=diffs.count)
    else:
        most_common_diff = sequence[1] - sequence[0]
    
    # Проверяем всю последовательность
    for i in range(len(sequence)-1):
        actual_diff = sequence[i+1] - sequence[i]
        expected_next = sequence[i] + most_common_diff
        
        if actual_diff != most_common_diff:
            errors.append({
                'position': i,
                'current': sequence[i],
                'expected_next': expected_next,
                'actual_next': sequence[i+1],
                'expected_diff': most_common_diff,
                'actual_diff': actual_diff
            })
    
    return errors, most_common_diff

def generate_arithmetic_sequence_with_start(model, vocab_size, start_sequence, length=8, temperature=1.0):
    """Генерация арифметической последовательности с начальными числами"""
    device = next(model.parameters()).device
    model.eval()
    
    # Начинаем с заданной последовательности
    sequence = start_sequence.copy()
    
    with torch.no_grad():
        for _ in range(length - len(start_sequence)):
            input_tensor = torch.tensor(sequence, device=device).unsqueeze(0)
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            
            # Применяем softmax для получения вероятностей
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Выбираем следующий токен на основе вероятностей
            next_token = torch.multinomial(probs, 1).item()
            
            sequence.append(next_token)
            
            if next_token >= vocab_size or len(sequence) >= length:
                break
    
    return sequence

def demonstrate_task1():
    """Демонстрация работы модели из задания 1"""
    print("=== Task 1 Demo: Word Generation ===")
    try:
        checkpoint = torch.load('models/task1_model.pth', map_location='cpu')
        
        model = CharTransformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['config']['d_model'],
            nhead=checkpoint['config']['nhead'],
            num_layers=checkpoint['config']['num_layers'],
            seq_len=checkpoint['config']['seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {checkpoint['vocab_size']}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Генерация нескольких примеров с разными параметрами
        print("\nGenerated words:")
        test_cases = [
            {"start": "a", "temp": 1.5, "top_k": 5, "top_p": 0.9},
            {"start": "b", "temp": 1.2, "top_k": 8, "top_p": 0.95},
            {"start": "c", "temp": 0.8, "top_k": 3, "top_p": 0.8},
            {"start": "ab", "temp": 1.0, "top_k": 6, "top_p": 0.9},
            {"start": "de", "temp": 1.3, "top_k": 4, "top_p": 0.85},
            {"start": "", "temp": 1.1, "top_k": 7, "top_p": 0.92},
        ]
        
        for i, params in enumerate(test_cases):
            word = generate_word_advanced(
                model, char2idx, idx2char, 
                start_chars=params["start"],
                temperature=params["temp"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                max_length=12
            )
            print(f"  {i+1}. '{word}' (start: '{params['start']}', temp: {params['temp']})")
            
    except Exception as e:
        print(f"Error loading Task 1 model: {e}")
        print("Make sure to run task_1.py first to train the model")

def demonstrate_task2():
    """Демонстрация работы модели из задания 2"""
    print("\n=== Task 2 Demo: Arithmetic Sequences ===")
    try:
        # Пробуем загрузить модель с разными размерами контекста
        context_sizes = [3, 5, 7, 9]
        available_models = []
        
        for ctx_size in context_sizes:
            model_path = f'models/task2_model_ctx{ctx_size}.pth'
            if os.path.exists(model_path):
                available_models.append(ctx_size)
        
        if not available_models:
            print("No trained models found for Task 2")
            print("Make sure to run task_2.py first to train the models")
            return
        
        print(f"Available context sizes: {available_models}")
        
        # Используем модель с самым большим контекстом
        best_ctx = max(available_models)
        checkpoint = torch.load(f'models/task2_model_ctx{best_ctx}.pth', map_location='cpu')
        
        model = ArithmeticTransformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['config']['d_model'],
            nhead=checkpoint['config']['nhead'],
            num_layers=checkpoint['config']['num_layers'],
            seq_len=checkpoint['config']['seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded successfully! (Context size: {checkpoint['config']['context_size']})")
        print(f"Vocabulary size: {checkpoint['vocab_size']}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Генерация нескольких арифметических последовательностей с начальными числами
        print("\nGenerated arithmetic sequences with analysis:")
        
        # Тестовые случаи с начальными последовательностями
        test_cases = [
            {"start": [2, 4], "length": 8, "temp": 0.3, "description": "Возрастающая +2"},
            {"start": [5, 10, 15], "length": 8, "temp": 0.3, "description": "Возрастающая +5"},  
            {"start": [1, 3, 5], "length": 8, "temp": 0.3, "description": "Возрастающая +2"},
            {"start": [10, 7, 4], "length": 8, "temp": 0.3, "description": "Убывающая -3"},
            {"start": [3, 6], "length": 8, "temp": 0.3, "description": "Возрастающая +3"},
            {"start": [20, 15, 10], "length": 8, "temp": 0.3, "description": "Убывающая -5"},
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Тест {i+1}: {test_case['description']} ---")
            print(f"Начальная последовательность: {test_case['start']}")
            
            sequence = generate_arithmetic_sequence_with_start(
                model, 
                checkpoint['vocab_size'], 
                start_sequence=test_case["start"],
                length=test_case["length"],
                temperature=test_case["temp"]
            )
            
            print(f"Полная последовательность: {sequence}")
            
            # Проверяем, является ли последовательность арифметической прогрессией
            is_arithmetic, difference = is_arithmetic_progression(sequence)
            
            if is_arithmetic:
                print(f"✅ Арифметическая прогрессия с разностью {difference}")
            else:
                print(f"❌ Не является арифметической прогрессией")
                
                # Детальный анализ ошибок
                errors, expected_diff = analyze_sequence_errors(sequence)
                if errors:
                    print(f"   Ожидаемая разность: {expected_diff}")
                    print(f"   Обнаружено ошибок: {len(errors)}")
                    for error in errors[:3]:  # Показываем первые 3 ошибки
                        print(f"   - На позиции {error['position']}: {error['current']} → {error['actual_next']} "
                              f"(ожидалось: {error['current']} → {error['expected_next']})")
            
            # Проверяем, правильно ли модель поняла начальную прогрессию
            start_is_arithmetic, start_diff = is_arithmetic_progression(test_case['start'])
            if start_is_arithmetic:
                continued_correctly = is_arithmetic and (difference == start_diff)
                if continued_correctly:
                    print(f"✅ Модель правильно продолжила прогрессию")
                else:
                    print(f"❌ Модель неправильно продолжила прогрессию")
            
            print("-" * 50)
            
    except Exception as e:
        print(f"Error loading Task 2 model: {e}")
        print("Make sure to run task_2.py first to train the model")

def demonstrate_task3():
    """Демонстрация работы модели из задания 3"""
    print("\n=== Task 3 Demo: Regularized Word Generation ===")
    try:
        # Пробуем загрузить разные конфигурации
        configs = [
            "Baseline",
            "Only_Dropout", 
            "Only_Weight_Tying",
            "Dropout_plus_Tying"
        ]
        
        available_configs = []
        for config in configs:
            model_path = f'models/task3_model_{config}.pth'
            if os.path.exists(model_path):
                available_configs.append(config)
        
        if not available_configs:
            print("No trained models found for Task 3")
            print("Make sure to run task_3.py first to train the models")
            return
        
        print(f"Available configurations: {available_configs}")
        
        # Используем модель с регуляризацией
        best_config = "Dropout_plus_Tying" if "Dropout_plus_Tying" in available_configs else available_configs[0]
        checkpoint = torch.load(f'models/task3_model_{best_config}.pth', map_location='cpu')
        
        model = RegularizedCharTransformer(
            vocab_size=checkpoint['vocab_size'],
            seq_len=checkpoint['config']['seq_len'],
            dropout_rate=checkpoint['config']['dropout'],
            tie_weights=checkpoint['config']['tie_weights'],
            d_model=checkpoint['config']['d_model'],
            nhead=checkpoint['config']['nhead'],
            num_layers=checkpoint['config']['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        
        print(f"Model loaded successfully! (Configuration: {checkpoint['config']['name']})")
        print(f"Vocabulary size: {checkpoint['vocab_size']}")
        if 'best_loss' in checkpoint:
            print(f"Best loss: {checkpoint['best_loss']:.4f}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Генерация с разными параметрами
        print("\nGenerated words with regularization:")
        test_cases = [
            {"start": "a", "temp": 1.5, "top_k": 5, "top_p": 0.9},
            {"start": "b", "temp": 1.2, "top_k": 8, "top_p": 0.95},
            {"start": "c", "temp": 0.8, "top_k": 3, "top_p": 0.8},
            {"start": "ab", "temp": 1.0, "top_k": 6, "top_p": 0.9},
            {"start": "de", "temp": 1.3, "top_k": 4, "top_p": 0.85},
            {"start": "", "temp": 1.1, "top_k": 7, "top_p": 0.92},
        ]
        
        for i, params in enumerate(test_cases):
            word = generate_word_advanced(
                model, char2idx, idx2char, 
                start_chars=params["start"],
                temperature=params["temp"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                max_length=12
            )
            print(f"  {i+1}. '{word}' (start: '{params['start']}', temp: {params['temp']})")
            
    except Exception as e:
        print(f"Error loading Task 3 model: {e}")
        print("Make sure to run task_3.py first to train the model")

def main():
    parser = argparse.ArgumentParser(description='Demonstrate trained Transformer models')
    parser.add_argument('task', type=int, choices=[1, 2, 3], 
                       help='Task number to demonstrate (1, 2, or 3)')
    
    args = parser.parse_args()
    
    if args.task == 1:
        demonstrate_task1()
    elif args.task == 2:
        demonstrate_task2()
    elif args.task == 3:
        demonstrate_task3()
    else:
        print("Invalid task number. Please choose 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()