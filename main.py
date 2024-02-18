from transformer import Transformer
from input_data import InputData
from input_embedding import InputEmbedding


def main():
    text = InputData('gerher eh erh er hre herherher', False)
    model = Transformer(text.data, text.sequence_len, 6)
    input = InputEmbedding(text.data, text.sequence_len, 512)
    print(input(text.data))
    
if __name__ == '__main__':
    main()