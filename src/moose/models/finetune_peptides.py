# direct reward backpropagation
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import pandas as pd

from moose.utils.finetune_utils import loss_wdce
from moose.utils.plotting import plot_data_with_distribution_seaborn, plot_data


def finetune(
    args,
    cfg,
    policy_model,
    reward_model,
    mcts=None,
    pretrained=None,
    filename=None,
    prot_name=None,
    eps=1e-5,
):
    """
    Finetuning with WDCE loss
    """
    # Enable anomaly detection to help debug in-place operation errors
    torch.autograd.set_detect_anomaly(True)

    base_path = args.base_path
    results_path = args.results_path
    dt = (1 - eps) / args.total_num_steps

    if args.no_mcts:
        assert pretrained is not None, "pretrained model is required for no mcts"
    else:
        assert mcts is not None, "mcts is required for mcts"

    # set model to train mode
    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # record metrics
    batch_losses = []
    # batch_rewards = []

    # initialize the final seqs and log_rnd of the trajectories that generated those seqs
    x_saved, log_rnd_saved, final_rewards_saved = None, None, None

    # Get score function names from reward_model
    score_func_names = reward_model.score_func_names

    # Initialize dynamic logging for scores
    valid_fraction_log = []
    score_logs = {name: [] for name in score_func_names}

    ### End of Fine-Tuning Loop ###
    pbar = tqdm(range(args.num_epochs))

    for epoch in pbar:
        # store metrics
        rewards = []
        losses = []

        policy_model.train()

        with torch.no_grad():
            if x_saved is None or epoch % args.resample_every_n_step == 0:
                # compute final sequences and trajectory log_rnd
                if args.no_mcts:
                    x_final, log_rnd, final_rewards = policy_model.sample_finetuned_with_rnd(
                        args, reward_model, pretrained
                    )
                else:
                    # decides whether to reset tree
                    if (epoch) % args.reset_every_n_step == 0:
                        x_final, log_rnd, final_rewards, _, _ = mcts.forward(resetTree=True)
                    else:
                        x_final, log_rnd, final_rewards, _, _ = mcts.forward(resetTree=False)

                # save for next iteration
                x_saved, log_rnd_saved, final_rewards_saved = x_final, log_rnd, final_rewards
            else:
                x_final, log_rnd, final_rewards = x_saved, log_rnd_saved, final_rewards_saved

        # compute wdce loss
        loss = loss_wdce(
            policy_model,
            log_rnd,
            x_final,
            num_replicates=args.wdce_num_replicates,
            centering=args.centering,
        )

        # gradient descent
        loss.backward()

        # optimizer
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)

        optim.step()
        optim.zero_grad()

        pbar.set_postfix(loss=loss.item())

        # sample a eval batch with updated policy to evaluate rewards
        x_eval, scores_dict, valid_fraction = policy_model.sample_finetuned(
            args, reward_model, batch_size=50, dataframe=False
        )

        # append to log dynamically
        valid_fraction_log.append(valid_fraction)
        for name in score_func_names:
            score_logs[name].append(scores_dict[name])

        batch_losses.append(loss.cpu().detach().numpy())

        losses.append(loss.cpu().detach().numpy())
        losses = np.array(losses)

        if args.no_mcts:
            mean_reward_search = final_rewards.mean().item()
            min_reward_search = final_rewards.min().item()
            max_reward_search = final_rewards.max().item()
            median_reward_search = final_rewards.median().item()
        else:
            mean_reward_search = np.mean(final_rewards)
            min_reward_search = np.min(final_rewards)
            max_reward_search = np.max(final_rewards)
            median_reward_search = np.median(final_rewards)

        # Dynamic print statement
        score_str = " ".join([f"{name} {scores_dict[name]:.4f}" for name in score_func_names])
        print(f"epoch {epoch} {score_str} mean loss {np.mean(losses):.4f}")

        # Dynamic wandb.log
        wandb_dict = {
            "epoch": epoch,
            "mean_loss": np.mean(losses),
            "mean_reward_search": mean_reward_search,
            "min_reward_search": min_reward_search,
            "max_reward_search": max_reward_search,
            "median_reward_search": median_reward_search,
        }
        wandb_dict.update({name: scores_dict[name] for name in score_func_names})
        wandb.log(wandb_dict)

        if (epoch + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f"model_{epoch}.ckpt")
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")

    ### End of Fine-Tuning Loop ###

    wandb.finish()

    # save logs and plot
    plot_path = f"{args.results_path}/{args.run_name}/plots"
    os.makedirs(plot_path, exist_ok=True)
    output_log_path = f"{args.results_path}/{args.run_name}/log_{filename}.csv"
    save_logs_to_file(valid_fraction_log, score_logs, output_path=output_log_path)

    plot_data(valid_fraction_log, save_path=f"{plot_path}/valid_{filename}.png")

    # Dynamic plotting for each score function
    for name in score_func_names:
        plot_data_with_distribution_seaborn(
            log1=score_logs[name],
            save_path=f"{plot_path}/{name}_{filename}.png",
            label1=f"Average {name.upper()} Score",
            title=f"Average {name.upper()} Score Over Iterations",
        )

    x_eval, scores_dict, valid_fraction, df = policy_model.sample_finetuned(
        args, reward_model, batch_size=200, dataframe=True
    )
    df.to_csv(f"{args.results_path}/{args.run_name}/{filename}_generation_results.csv", index=False)
    print(
        f"generation results saved at {args.results_path}/{args.run_name}/{filename}_generation_results.csv"
    )
    print(f"plots saved at {plot_path}")
    print(f"logs saved at {output_log_path}")

    return batch_losses


def save_logs_to_file(valid_fraction_log, score_logs, output_path):
    """
    Saves the logs (valid_fraction_log and score logs) to a CSV file.

    Parameters:
        valid_fraction_log (list): Log of valid fractions over iterations.
        score_logs (dict): Dictionary of score logs, keyed by score function names.
        output_path (str): Path to save the log CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Combine logs into a DataFrame
    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
    }
    # Add score logs dynamically
    for name, log in score_logs.items():
        log_data[name] = log

    df = pd.DataFrame(log_data)

    # Save to CSV
    df.to_csv(output_path, index=False)
