import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--task', type=str, default='classification', choices=["classification", "recsys", "llm_sft"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["ml1m", "cifar10", "yahoo", "aya"])
    parser.add_argument(
        "--noniid",
        type=str,
        default="dirichlet-user-genre",
        choices=[
            "dirichlet-user-genre",
            "dir_user_genre",
            "overlap-dirichlet-user-genre",
            "overlap_dir_user_genre",
            "group",
            "user",
            "iid",
            "noniid",
            "homo",
            "lang-overlap",
            "schemeA",
            "dirichlet-user-genre",
        ],
        help="Data partition mode. Use overlap-dirichlet-user-genre for overlapped users across clients."
    )
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--personalized_learning_rate', type=float, default=0.01,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=50, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=42, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="../Data/data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default=2,
                        help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')

    # ====== recsys ======
    parser.add_argument("--aggregate_user_overlap", action="store_true", default=True, help="If set, aggregate user embedding rows only for users that appear in multiple clients.")
    parser.add_argument("--recsys_model", type=str, default="ncf", choices=["mf", "lightgcn"])
    parser.add_argument('--embed_dim', type=int, default=64, help='embed_dim')
    parser.add_argument('--gcn_layers', type=int, default=2, help='gcn_layers')
    parser.add_argument('--pos_threshold', type=int, default=4, help='pos_threshold')
    parser.add_argument('--num_neg', type=int, default=100, help='num_neg')
    parser.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--ranking_max_users", type=int, default=None)
    parser.add_argument('--rank_lambda', type=float, default=0.1)
    parser.add_argument("--force_top_genres_k", type=int, default=10,
                        help="Force keeping top-K genres by frequency. Applied to both train/test filtering.")
    parser.add_argument("--force_top_genres_source", type=str, default="all", choices=["all", "train", "test"],
                        help="Frequency source used to pick top-K genres: all/train/test.")
    parser.add_argument("--use_user_bind", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable user-bind mechanism (and allow disabling via --no-use_user_bind).")
    parser.add_argument("--user_bind_k", type=int, default=3,
                        help="User-bind: bind each user to top-k candidate clients.")
    parser.add_argument("--user_bind_sharpness", type=float, default=5.0,
                        help="User-bind: sharpness for user->client binding distribution.")
    parser.add_argument("--user_bind_pop_tau", type=float, default=1.0,
                        help="User-bind: temperature/weight for popularity prior (if used).")
    parser.add_argument("--user_min_train_inter", type=int, default=2,
                        help="Minimum number of training interactions per user.")
    parser.add_argument("--clients_per_genre", type=int, default=2,
                        help="How many clients are assigned to each genre pool.")
    parser.add_argument("--assign_sharpness", type=float, default=4,
                        help="Sharpness for assigning interactions/users to clients (genre-pool assignment).")
    parser.add_argument("--min_interactions_per_client", type=int, default=200,
                        help="Minimum interactions per client; re-sample assignment if violated.")
    parser.add_argument("--max_retries", type=int, default=30,
                        help="Maximum retries to satisfy min_interactions_per_client constraint.")

    # ====== LLM SFT ======
    parser.add_argument("--aya_top_k_langs", type=int,
                        default=10)
    parser.add_argument("--aya_lang_codes", type=str, nargs="*", default=None,
                        help="Optional manual override for language_code list. "
                             "If None, auto-pick top-K by frequency from train split.")
    parser.add_argument("--min_langs_per_client", type=int, default=2,
                        help="Min number of languages per client (default=2).")
    parser.add_argument("--max_langs_per_client", type=int, default=4,
                        help="Max number of languages per client (default=4).")
    parser.add_argument("--avg_langs_per_client", type=int, default=3,
                        help="Average number of languages per client when constructing overlaps (default=3).")
    parser.add_argument("--llm_max_length", type=int, default=512,
                        help="Max sequence length for SFT tokenization (default=1024).")
    parser.add_argument("--model_name_or_path", type=str, default="model/Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model id or local path.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Allow loading models with remote code if needed.")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantized loading for QLoRA (bitsandbytes).")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="*", default=["q_proj", "v_proj", "o_proj"],
                        help="Optional override. Example: q_proj k_proj v_proj o_proj. "
                             "If None, auto-guess common modules.")
    parser.add_argument("--llm_lr", type=float, default=5e-5,
                        help="Learning rate for LoRA params in SFT.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps for LLM SFT.")
    parser.add_argument("--aya_balance_per_lang", type=int, default=2000,
                        help="Per-language cap for balancing Top-K languages.")
    parser.add_argument("--aya_dirichlet_beta", type=float, default=0.3,
                        help="Dirichlet beta for non-IID partition.")


    args = parser.parse_args()
    cfg = dict()
    if args.dataset in {'cifar10', 'yahoo'}:
        cfg['classes_size'] = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args, cfg

