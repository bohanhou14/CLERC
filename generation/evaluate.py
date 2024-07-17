from argparse import ArgumentParser

from legal_generation.evaluation.legal_eval import LegalEval


def main():
    parser = ArgumentParser()
    parser.add_argument('root', type=str, help='root path of predictions')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bart-path')
    parser.add_argument('--n-example', type=int, default=100)
    args = parser.parse_args()
    legal_eval = LegalEval(**vars(args))
    legal_eval.run()


if __name__ == '__main__':
    main()
