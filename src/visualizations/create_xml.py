import argparse
from os.path import join as pathjoin
import pandas as pd
import pickle
import lime
import shap
from pdpbox import pdp
import matplotlib.pyplot as plt



def load_model(model_filename):
    return pickle.load(open(model_filename, "rb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xr", "--xtrain_resampled_path", help="X_train_resampled file path")
    parser.add_argument("-xt", "--x_test_path", help="X_test file path ")
    parser.add_argument("-s", "--saved_model_path", help="Filepath to saved trained model", required=True)
    parser.add_argument("-ls", "--lime_seed", type=int, help="Seed for Lime")
    parser.add_argument("-lo", "--lime_output_path", help="Output path to save Lime Explanation")
    parser.add_argument("-p1", "--pdp_feature_1", help="Feature 1 for pdp Plot")
    parser.add_argument("-p2", "--pdp_feature_2", help="Feature 2 for pdp Plot")
    parser.add_argument("-pdp", "--pdp_dependence_path", help="Output path for pdp dependence output")
    parser.add_argument("-pi", "--pdp_isolate_var", help="Feature to run pdpIsolate on")
    parser.add_argument("-pip", "--pdp_isloate_path", help="Output path for pdp isolate output")
    parser.add_argument("-sdp", "--shap_dot_path", help="Path for shap dot chart")
    parser.add_argument("-sbp", "--shap_bar_path", help="Path for shap bar chart")
    
    args = parser.parse_args()
    if not all((args.pdp_feature_1, args.pdp_feature_2)):
        args.pdp_feature_1 = None
        args.pdp_feature_2 = None
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

def create_xml():
    model = load_model(parsed_args.get("saved_model_path"))
    DEAFULT_XTRAIN_RESAMPLED_PATH = pathjoin("..","..","data","processed","X_train_resampled.csv")
    Xt_resampled = pd.read_csv(parsed_args.get("xtrain_resampled_path",DEAFULT_XTRAIN_RESAMPLED_PATH))
    DEFAULT_XTEST_PATH = pathjoin("..","..","data","processed","X_test.csv")
    X_test = pd.read_csv(parsed_args.get("x_test_path", DEFAULT_XTEST_PATH))
    trained_columns = model.feature_names_in_.tolist()
    X_test = X_test[trained_columns]
    figure_path = pathjoin("..","..","reports","figures")
    # LIME
    print("============= Plotting Lime Plot =============")    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(Xt_resampled.values, mode='classification', feature_names=trained_columns, random_state=parsed_args.get("lime_seed",42))
    lime_exp = lime_explainer.explain_instance(X_test.values[0], model.predict_proba, num_features=25)
    print("============= Saving Lime Plot =============")    
    DEFAULT_LIME_OUTPUT_PATH = pathjoin(figure_path,"lime.html")
    lime_exp.save_to_file(parsed_args.get("lime_output_path", DEFAULT_LIME_OUTPUT_PATH))

    print("============= Plotting Partial Dependence Plot =============")    
    # Partial dependence plots pt1
    DEFAULT_PDP_DEPENDENCE_FEATURES = ['phone_home_valid', 'phone_mobile_valid']
    pdp_dependence_features = [parsed_args.get("pdp_feature_1"),parsed_args.get("pdp_feature_2")] if parsed_args.get("pdp_feature_1") else DEFAULT_PDP_DEPENDENCE_FEATURES
    interact_features = pdp.PDPInteract(model=model, df=X_test.copy(), model_features=trained_columns, 
                            features=pdp_dependence_features, feature_names=pdp_dependence_features, n_classes=2)

    fig, axes = interact_features.plot(plot_type='contour', engine="matplotlib")
    print("============= Saving Partial Dependence Plot =============")    
    DEFAULT_PDP_DEPENDENCE_PATH = pathjoin(figure_path,"pdp_dependence.png")
    fig.savefig(parsed_args.get("pdp_dependence_path",DEFAULT_PDP_DEPENDENCE_PATH))
    # fig.write_image(file=parsed_args.get("pdp_dependence_path",DEFAULT_PDP_DEPENDENCE_FEATURES), format=".png")

    print("============= Plotting Partial Isolated Plot =============")    
    DEFAULT_PDP_ISOLATE_FEATURE = "credit_risk_score"
    pdp_isolate_var = parsed_args.get("pdp_isolate_var", DEFAULT_PDP_ISOLATE_FEATURE)
    # PDP pt2 credit_risk_score
    pdp_isolate = pdp.PDPIsolate(model=model, df=X_test.copy(), model_features=trained_columns,
                feature=pdp_isolate_var, feature_name=pdp_isolate_var, n_classes=2)
    fig, axes = pdp_isolate.plot(plot_pts_dist=True,to_bins=False, engine="matplotlib")
    print("============= Saving Partial Isolated Plot =============") 
    DEFAULT_PDP_ISOLATE_PATH = pathjoin(figure_path,"pdp_isolate.png")
    # fig.write_image(file=parsed_args.get("pdp_isloate_path",DEFAULT_PDP_ISOLATE_PATH), format=".png")
    fig.savefig(parsed_args.get("pdp_isloate_path", DEFAULT_PDP_ISOLATE_PATH))
    
    print("============= Plotting Shap dot Plots =============")    
    # # SHAP may have errors: must catch
    try:
        shap_explainer = shap.Explainer(model, Xt_resampled)
        shap_values = shap_explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        print("============= Saving Shap dot Plot =============")    
        DEFAULT_SHAP_DOT_OUTPUT_PATH = pathjoin(figure_path,"shap_dot_plot.png")
        plt.savefig(parsed_args.get("shap_dot_path", DEFAULT_SHAP_DOT_OUTPUT_PATH))
        plt.cla()
        print("============= Plotting Shap bar Plots =============")    
        shap_explainer = shap.Explainer(model, Xt_resampled)
        shap_values = shap_explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        print("============= Saving Shap bar Plot =============")  
        DEFAULT_SHAP_BAR_OUTPUT_PATH = pathjoin(figure_path,"shap_bar_plot.png")
        plt.savefig(parsed_args.get("shap_bar_path", DEFAULT_SHAP_BAR_OUTPUT_PATH))
    except TypeError:
        print(TypeError("Model not supported by SHAP"))


if __name__ == "__main__":
    parsed_args = parse_arguments()
    create_xml()