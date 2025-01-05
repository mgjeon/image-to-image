def get_fake_target(model, inputs):
    fake_targets = model(inputs)
    return fake_targets