import asyncio
import argparse
from training.main import main as training


def main() -> None:
    parser = argparse.ArgumentParser(description = "Tebakaja Model Trainer")

    parser.add_argument('-a', '--algorithm', type = str, required = True,
        help = 'select the algorithm to be trained (LSTM, GRU, LSTM_GRU)')

    parser.add_argument('-e', '--epochs', type = int, required = True, help = 'epochs')
    parser.add_argument('-b', '--batchs', type = int, required = True, help = 'batch length')
    parser.add_argument('-s', '--sequences', type = int, required = True, help = 'sequences length')

    args = parser.parse_args()
    event_loop = asyncio.get_event_loop()

    event_loop.run_until_complete(
        training(
            epochs     = args.epochs,
            batch_size = args.batchs,
            algorithm  = args.algorithm,
            sequence_length = args.sequences
        )
    )


if __name__ == "__main__": main()

