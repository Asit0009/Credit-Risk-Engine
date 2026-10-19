import pytest
from streamlit.testing.v1 import AppTest


def test_app_startup_no_exception():
    at = AppTest.from_file("app.py", default_timeout=30).run()  # start app
    assert not at.exception
    assert len(at.title) > 0  # check title exists
    assert "Credit Risk Assessment" in at.title[0].value


def test_app_navigation_views():
    at = AppTest.from_file("app.py", default_timeout=30).run()

    # check switching between sidebar tabs
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("Decision History").run()
        assert not at.exception
        assert any("Decision History" in t.value for t in at.title)

        at.sidebar.radio[0].set_value("Model Performance").run()
        assert not at.exception
        assert any("Model Performance" in t.value for t in at.title)

        at.sidebar.radio[0].set_value("About Project").run()
        assert not at.exception
        assert any("About Project" in t.value for t in at.title)


def test_app_assessment_submission():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception

    # find assess button
    assess_btn = None
    for btn in at.button:
        if "Assess Credit Risk" in btn.label:
            assess_btn = btn
            break

    assert assess_btn is not None
    assess_btn.click().run()  # simulate button click
    assert not at.exception

    # verify results rendered
    assert len(at.info) > 0
    assert any("Assessment Result" in m.value for m in at.markdown)
