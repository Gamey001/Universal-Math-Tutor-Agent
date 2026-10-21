from tutor.images import detect_image_issue, prepare_for_model


def test_blank_photo_is_flagged(blank_photo):
    assert detect_image_issue(blank_photo) == "blank"


def test_photo_with_content_passes(text_photo, blurry_photo):
    assert detect_image_issue(text_photo) is None
    assert detect_image_issue(blurry_photo) is None  # blurry but not empty: let the models try


def test_non_image_is_unopenable():
    assert detect_image_issue(b"definitely not a picture") == "unopenable"


def test_prepare_downscales(text_photo):
    from PIL import Image
    import io
    out = Image.open(io.BytesIO(prepare_for_model(text_photo)))
    assert max(out.size) <= 1280
