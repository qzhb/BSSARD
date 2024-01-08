import argparse

def get_argsparser():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--save_dir', type=str, default='datasets_t7', help='path to save processed dataset')
    parser.add_argument('--task', type=str, default='charades', help='target task')
    parser.add_argument('--vf', type=str, default='org', help='[i3d | c3d] for visual features')
    parser.add_argument('--qf', type=str, default='org', help='[org | cd ] for query features')
    parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
    # model parameters
    parser.add_argument("--word_size", type=int, default=None, help="number of words")
    parser.add_argument("--char_size", type=int, default=None, help="number of characters")
    parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
    parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
    parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
    parser.add_argument("--dim", type=int, default=128, help="hidden size")
    parser.add_argument("--highlight_lambda", type=float, default=5.0, help="lambda for highlight region")
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
    parser.add_argument('--predictor', type=str, default='transformer', help='[rnn | transformer]')
    # training/evaluation parameters
    parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--mode", type=str, default="train", help="[train | test]")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
    parser.add_argument("--extend", type=float, default=0.1, help="highlight region extension")
    parser.add_argument("--period", type=int, default=100, help="training loss print period")
    parser.add_argument('--model_dir', type=str, default='results', help='path to save trained model weights')
    parser.add_argument('--model_name', type=str, default='base', help='model name')
    parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
    # my args
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--checkpoint_dir', default="", help="load checkpoint dir")
    parser.add_argument('--m', type=int, default=1, help="generator train times")
    parser.add_argument('--times', type=int, default=1, help="xcqa times")
    parser.add_argument('--loss_type', type=str, default='', help="")
    parser.add_argument('--train_strategy', type=str, default='alter', help="")
    parser.add_argument('--save_num_counts', type=bool, default=False)
    parser.add_argument('--schedule', type=str, default="linear")

    return parser