from ModelRunner import ModelRunner


def run(model_generator, optimizer_generator, params, accuracy, train_loader, valid_loader,
        criterion=None, device=None, n_epochs=5):
    best_model = None
    best_valid_metric = 0
    best_parameter_value = params[0]

    for p in params:
        print(f"Trying {p}")
        model = model_generator(p)
        optimizer = optimizer_generator(model)
        history = ModelRunner(model, accuracy, optimizer, criterion, device) \
                   .train_model(train_loader, valid_loader, n_epochs=n_epochs)
        if history.best_valid_metric > best_valid_metric:
            best_valid_metric = history.best_valid_metric
            best_model = model
            best_parameter_value = p

    return best_model, best_valid_metric, best_parameter_value