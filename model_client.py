def generate_answer(
    prompt,
    client,
    model,
):
    if not prompt.strip():
        raise ValueError(
            "prompt must not be empty"
        )

    if not model.strip():
        raise ValueError(
            "model must not be empty"
        )

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    answer_text = response.output_text.strip()

    if not answer_text:
        raise ValueError(
            "model response must not be empty"
        )

    return answer_text
