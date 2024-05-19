from environment.stock_trading_env import StockTradingEnv
from environment.aec_wrapper.autoencoder_wrapper import AecObservationWrapper
from environment.gnn_wrapper.gnn_wrapper import GnnObservationWrapper

def make_env(args):
    gnn_hidden_channels = [int(x) for x in args.gnn_hidden_channels.split(',')]
    env = StockTradingEnv(args.online,args.trading_window_size,args.buffer_size,start_date = args.train_start_date,end_date=args.train_end_date,assets_number=args.assets_number,starting_cash=args.starting_cash)
    env = AecObservationWrapper(env,aec_layers_size = [5,[10,8]],out_features = 3,autoencoder_path='E:/baselines/DeepPocket-master/environment/aec_wrapper/autoencoder/linear_autoencoder_sp500.pt')
    env = GnnObservationWrapper(env,args.trading_window_size,args.gnn_input_channels,gnn_hidden_channels,args.gnn_output_channels,args.cheb_k,number_of_assets=args.assets_number)

    return env