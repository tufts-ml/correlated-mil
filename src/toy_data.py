import argparse
import os
import pandas as pd
# PyTorch
import torch
# Importing our custom module(s)
import losses
import models
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="toy_data.py")
    parser.add_argument("--alpha", default=0.0, help="TODO (default: 0.0)", type=float)
    parser.add_argument("--beta", default=0.0, help="TODO (default: 0.0)", type=float)
    parser.add_argument("--batch_size", default=64, help="Batch size (default: 64)", type=int)
    parser.add_argument("--criterion", default="ERM", help="TODO (default: \"ERM\")", type=str)
    parser.add_argument("--data_seed_test", default=2, help="TODO (default: 2)", type=int)
    parser.add_argument("--data_seed_train", default=0, help="TODO (default: 0)", type=int)
    parser.add_argument("--data_seed_val", default=1, help="TODO (default: 1)", type=int)
    parser.add_argument("--delta", default=1.0, help="TODO (default: 1.0)", type=float)
    parser.add_argument("--deltaS", default=1, help="TODO (default: 1)", type=int)
    parser.add_argument("--embedding_level", action='store_true', default=False, help='Whether or not to use the embedding-level approach (default: False)')
    parser.add_argument("--epochs", default=1000, help="Number of epochs (default: 1000)", type=int)
    parser.add_argument("--experiments_directory", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--instance_conv", action='store_true', default=False, help='Whether or not to use instance convolutions (default: False)')
    parser.add_argument("--kernel_size", default=3, help="Kernel size for instance convolutions (default: 3)", type=int)
    parser.add_argument("--lr", default=0.01, help="Learning rate (default: 0.01)", type=float)
    parser.add_argument("--model_name", default="test", help="Model name (default: \"test\")", type=str)
    parser.add_argument("--N_test", default=100, help="Number of testing samples (default: 100)", type=int)
    parser.add_argument("--N_train", default=800, help="Number of training samples (default: 800)", type=int)
    parser.add_argument("--N_val", default=100, help="Number of validation samples (default: 100)", type=int)
    parser.add_argument("--pooling", default="max", help="Pooling operation (default: \"max\")", type=str)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument("--seed", default=42, help="TODO (default: 42)", type=int)
    parser.add_argument("--use_pos_embedding", action='store_true', default=False, help='Whether or not to use positional embedding (default: False)')
    parser.add_argument("--weight_decay", default=0.0, help="Weight decay (default: 0.0)", type=float)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    os.makedirs(args.experiments_directory, exist_ok=True)

    X_train, lengths_train, u_train, y_train = utils.generate_toy_data(args.N_train, delta=args.delta, deltaS=args.deltaS, seed=args.data_seed_train)
    X_val, lengths_val, u_val, y_val = utils.generate_toy_data(args.N_val, delta=args.delta, deltaS=args.deltaS, seed=args.data_seed_val)
    X_test, lengths_test, u_test, y_test = utils.generate_toy_data(args.N_test, delta=args.delta, deltaS=args.deltaS, seed=args.data_seed_test)
    
    train_dataset = utils.ToyDataset(X_train, lengths_train, y_train)
    val_dataset = utils.ToyDataset(X_val, lengths_val, y_val)
    test_dataset = utils.ToyDataset(X_test, lengths_test, y_test)
    
    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if args.embedding_level:
        model = models.PoolClf(in_features=768, out_features=1, pooling=args.pooling, use_pos_embedding=args.use_pos_embedding)
    else:
        model = models.ClfPool(in_features=768, out_features=1, instance_conv=args.instance_conv, kernel_size=args.kernel_size, pooling=args.pooling, use_pos_embedding=args.use_pos_embedding)
    model.to(device)
        
    if args.criterion == "ERM":
        criterion = losses.ERMLoss(criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "L1":
        criterion = losses.L1Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "L2":
        criterion = losses.L2Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "GuidedL1":
        criterion = losses.GuidedAttentionL1Loss(alpha=args.alpha, beta=args.beta, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "GuidedNormalL1":
        criterion = losses.GuidedNormalL1Loss(alpha=args.alpha, beta=args.beta, criterion=torch.nn.BCEWithLogitsLoss())
    else:
        raise NotImplementedError(f"The specified criterion \"{self.criterion}\" is not implemented.")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    columns = ["epoch", "test_acc", "test_auroc", "test_auprc", "test_loss", "test_nll", "train_acc", "train_auroc", "train_auprc", "train_loss", "train_nll", "val_acc", "val_auroc", "val_auprc", "val_loss", "val_nll"]
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        
        shuffled_train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
        #train_metrics = utils.evaluate(model, criterion, train_loader)
        train_metrics = shuffled_train_metrics
        val_metrics = utils.evaluate(model, criterion, val_loader)
        test_metrics = utils.evaluate(model, criterion, test_loader)
        
        row = [epoch, test_metrics["acc"], test_metrics["auroc"], test_metrics["auprc"], test_metrics["loss"], test_metrics["nll"], train_metrics["acc"], train_metrics["auroc"], train_metrics["auprc"], train_metrics["loss"], train_metrics["nll"], val_metrics["acc"], val_metrics["auroc"], val_metrics["auprc"], val_metrics["loss"], val_metrics["nll"]]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f"{args.experiments_directory}/{args.model_name}.csv")
    
        if args.save and epoch == model_history_df.val_auroc.idxmax():
            torch.save(model.state_dict(), f"{args.experiments_directory}/{args.model_name}.pt")
