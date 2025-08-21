import argparse
import pandas as pd
from pathlib import Path


def main(input_network, output_network):
    Path(output_network).parent.mkdir(exist_ok=True, parents=True)
    pd.read_csv(input_network, sep="\\s+", header=None, usecols=[0, 1]).to_csv(
        output_network, header=False, index=False, sep="\t"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a network file.")
    parser.add_argument(
        "--input-network", required=True, help="Path to the input network file."
    )
    parser.add_argument(
        "--output-network", required=True, help="Path to the output network file."
    )

    args = parser.parse_args()
    main(args.input_network, args.output_network)
