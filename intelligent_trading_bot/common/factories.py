from intelligent_trading_bot.pipeline.components.model_operator import ModelOperator

def create_component(definition: dict):
    type_ = definition["type"]
    params = definition["params"]

    if type_ == "ModelOperator":
        return ModelOperator(**params)

    raise ValueError(f"Unknown component type: {type_}")
