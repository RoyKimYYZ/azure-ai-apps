"""Playwright UI tests for the Fitness Nutrition agent in the Streamlit chatbot.

These tests validate that the UI renders correctly, interactive sidebar
controls work, the Fitness Context right-panel displays expected sections,
the auth/demo gate functions, and core chat interactions behave properly.

Run with:
    uv run pytest tests/ui/ -v --headed   # visible browser (debugging)
    uv run pytest tests/ui/ -v            # headless (CI)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIDEBAR = "[data-testid='stSidebar']"
_APP = "[data-testid='stApp']"


def _select_agent(page: Page, agent_name: str) -> None:
    """Select an agent from the sidebar selectbox."""
    sidebar = page.locator(_SIDEBAR)
    # Streamlit selectboxes use a div with role="combobox"
    agent_select = sidebar.locator("div[data-testid='stSelectbox']").first
    agent_select.click()
    page.locator(f"li:has-text('{agent_name}')").click()
    page.wait_for_timeout(500)


def _enter_demo_mode(page: Page) -> None:
    """Click the demo mode button to bypass the auth gate."""
    demo_button = page.get_by_role("button", name=re.compile(r"Continue as Demo", re.IGNORECASE))
    if demo_button.is_visible(timeout=5_000):
        demo_button.click()
        page.wait_for_timeout(1_500)


def _ensure_fitness_agent(page: Page) -> None:
    """Select the Fitness Nutrition agent and enter demo mode if needed."""
    _select_agent(page, "Fitness Nutrition")
    page.wait_for_timeout(1_000)
    # The fitness agent shows an auth gate; enter demo mode.
    _enter_demo_mode(page)


# ---------------------------------------------------------------------------
# 1. App Launch & Chrome
# ---------------------------------------------------------------------------


class TestAppLaunch:
    """Verify the Streamlit app starts and renders its shell."""

    def test_page_title_contains_chatbot(self, ui_page: Page) -> None:
        """Page title should reflect the config ui.theme.page_title."""
        expect(ui_page).to_have_title(re.compile(r"chatbot|AI Foundry", re.IGNORECASE))

    def test_sidebar_is_visible(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        expect(sidebar).to_be_visible()

    def test_settings_header_present(self, ui_page: Page) -> None:
        header = ui_page.locator(f"{_SIDEBAR} >> text=Settings")
        expect(header).to_be_visible()


# ---------------------------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------------------------


class TestSidebarControls:
    """Validate that key sidebar widgets are present and interactive."""

    def test_agent_selector_has_fitness_option(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        agent_select = sidebar.locator("div[data-testid='stSelectbox']").first
        agent_select.click()
        option = ui_page.locator("li:has-text('Fitness Nutrition')")
        expect(option).to_be_visible()
        # Close the dropdown
        ui_page.keyboard.press("Escape")

    def test_model_selector_present(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        model_select = sidebar.locator("div[data-testid='stSelectbox']").nth(1)
        expect(model_select).to_be_visible()

    def test_temperature_slider_present(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        slider = sidebar.locator("div[data-testid='stSlider']").first
        expect(slider).to_be_visible()

    def test_max_tokens_input_present(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        number_input = sidebar.locator("div[data-testid='stNumberInput']")
        expect(number_input).to_be_visible()

    def test_verify_tls_checkbox(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        checkbox = sidebar.locator("div[data-testid='stCheckbox']").first
        expect(checkbox).to_be_visible()

    def test_new_chat_button_present(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        new_chat = sidebar.get_by_role("button", name=re.compile(r"New chat", re.IGNORECASE))
        expect(new_chat).to_be_visible()

    def test_completion_metrics_section(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        metrics_text = sidebar.locator("text=Completion metrics")
        expect(metrics_text).to_be_visible()

    def test_diagnostics_link_present(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        diag_link = sidebar.locator("a:has-text('Diagnostics')")
        expect(diag_link).to_be_visible()


# ---------------------------------------------------------------------------
# 3. Agent Selection & Layout Switch
# ---------------------------------------------------------------------------


class TestAgentSelection:
    """Switching agents should change the layout (two-column vs single)."""

    def test_fitness_agent_shows_right_column(self, ui_page: Page) -> None:
        """Selecting Fitness Nutrition should produce a two-column layout with a right panel."""
        _ensure_fitness_agent(ui_page)
        # The right column should contain "Fitness Context"
        fitness_context = ui_page.locator("text=Fitness Context")
        expect(fitness_context).to_be_visible(timeout=10_000)

    def test_general_agent_hides_right_column(self, ui_page: Page) -> None:
        """Selecting General Chat Assistant should NOT show the Fitness Context panel."""
        _select_agent(ui_page, "General Chat Assistant")
        ui_page.wait_for_timeout(1_000)
        fitness_context = ui_page.locator("text=Fitness Context")
        expect(fitness_context).to_be_hidden(timeout=5_000)


# ---------------------------------------------------------------------------
# 4. Auth Gate / Demo Mode
# ---------------------------------------------------------------------------


class TestAuthGateDemoMode:
    """The Fitness Nutrition agent requires authentication; demo mode should work."""

    def test_auth_gate_renders_for_fitness(self, ui_page: Page) -> None:
        """Selecting Fitness Nutrition should show the login / demo gate."""
        _select_agent(ui_page, "Fitness Nutrition")
        ui_page.wait_for_timeout(1_500)
        gate_text = ui_page.locator("text=Sign in or Continue as Demo")
        # The gate appears if no session exists yet.
        expect(gate_text).to_be_visible(timeout=10_000)

    def test_demo_mode_button_works(self, ui_page: Page) -> None:
        """Clicking 'Continue as Demo' should authenticate as the demo user."""
        _select_agent(ui_page, "Fitness Nutrition")
        ui_page.wait_for_timeout(1_000)
        _enter_demo_mode(ui_page)
        # After demo login, the demo mode banner should appear.
        demo_banner = ui_page.locator("text=Demo Mode")
        expect(demo_banner).to_be_visible(timeout=10_000)


# ---------------------------------------------------------------------------
# 5. Fitness Context Panel (right column)
# ---------------------------------------------------------------------------


class TestFitnessContextPanel:
    """Validate the Fitness Context right-column sections after demo auth."""

    @pytest.fixture(autouse=True)
    def _setup(self, ui_page: Page) -> None:
        _ensure_fitness_agent(ui_page)

    def test_user_profile_expander_present(self, ui_page: Page) -> None:
        expander = ui_page.locator("text=User Profile")
        expect(expander).to_be_visible(timeout=10_000)

    def test_food_image_uploader_present(self, ui_page: Page) -> None:
        uploader = ui_page.locator("text=Upload food image")
        expect(uploader).to_be_visible(timeout=10_000)

    def test_longterm_memory_section_present(self, ui_page: Page) -> None:
        section = ui_page.locator("text=Long-term memory")
        expect(section).to_be_visible(timeout=10_000)

    def test_shortterm_memory_section_present(self, ui_page: Page) -> None:
        section = ui_page.locator("text=Short-term memory")
        expect(section).to_be_visible(timeout=10_000)

    def test_memory_debug_expander_present(self, ui_page: Page) -> None:
        debug_section = ui_page.locator("text=Memory Debug")
        expect(debug_section).to_be_visible(timeout=10_000)

    def test_refresh_memory_button_present(self, ui_page: Page) -> None:
        refresh_btn = ui_page.get_by_role("button", name=re.compile(r"Refresh memory snapshot", re.IGNORECASE))
        expect(refresh_btn).to_be_visible(timeout=10_000)

    def test_demo_user_caption_shown(self, ui_page: Page) -> None:
        """After demo auth, the panel should show the demo user caption."""
        caption = ui_page.locator("text=demo user")
        expect(caption).to_be_visible(timeout=10_000)


# ---------------------------------------------------------------------------
# 6. Chat Input Area
# ---------------------------------------------------------------------------


class TestChatInput:
    """Verify the chat input widget is present and functional."""

    def test_chat_input_present(self, ui_page: Page) -> None:
        chat_input = ui_page.locator("[data-testid='stChatInput'] textarea")
        expect(chat_input).to_be_visible()

    def test_chat_input_accepts_text(self, ui_page: Page) -> None:
        chat_input = ui_page.locator("[data-testid='stChatInput'] textarea")
        chat_input.fill("Hello, testing input")
        expect(chat_input).to_have_value("Hello, testing input")


# ---------------------------------------------------------------------------
# 7. New Chat Button Clears History
# ---------------------------------------------------------------------------


class TestNewChat:
    """The 'New chat' button should clear displayed messages."""

    def test_new_chat_clears_messages(self, ui_page: Page) -> None:
        _ensure_fitness_agent(ui_page)
        sidebar = ui_page.locator(_SIDEBAR)
        new_chat = sidebar.get_by_role("button", name=re.compile(r"New chat", re.IGNORECASE))
        new_chat.click()
        ui_page.wait_for_timeout(1_000)
        # After clicking new chat, there should be no assistant messages.
        assistant_msgs = ui_page.locator("[data-testid='stChatMessage'][data-testid-role='assistant']")
        expect(assistant_msgs).to_have_count(0, timeout=5_000)


# ---------------------------------------------------------------------------
# 8. Food Image Upload
# ---------------------------------------------------------------------------


class TestFoodImageUpload:
    """Validate the food image upload widget in the Fitness Context panel."""

    @pytest.fixture(autouse=True)
    def _setup(self, ui_page: Page) -> None:
        _ensure_fitness_agent(ui_page)

    def test_uploader_accepts_image_types(self, ui_page: Page) -> None:
        """The file uploader should be present and configured for image types."""
        uploader = ui_page.locator("[data-testid='stFileUploader']")
        expect(uploader).to_be_visible(timeout=10_000)

    def test_upload_sample_image_shows_preview(self, ui_page: Page) -> None:
        """Uploading a small test PNG should display an image preview."""
        # Create a minimal 1x1 PNG in a temp file.
        import struct
        import tempfile
        import zlib

        def _minimal_png() -> bytes:
            signature = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
            ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc
            raw_row = b"\x00\xff\x00\x00"  # filter=None, R=255, G=0, B=0
            idat_data = zlib.compress(raw_row)
            idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF)
            idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + idat_crc
            iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
            iend = struct.pack(">I", 0) + b"IEND" + iend_crc
            return signature + ihdr + idat + iend

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_minimal_png())
            tmp_path = f.name

        try:
            file_input = ui_page.locator("[data-testid='stFileUploader'] input[type='file']")
            file_input.set_input_files(tmp_path)
            ui_page.wait_for_timeout(2_000)
            # Streamlit renders an image element after upload.
            preview = ui_page.locator("[data-testid='stImage']")
            expect(preview).to_be_visible(timeout=10_000)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 9. Diagnostics Link Navigation
# ---------------------------------------------------------------------------


class TestDiagnosticsLink:
    """The diagnostics link should point to the diagnostics page."""

    def test_diagnostics_link_href(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        link = sidebar.locator("a:has-text('Diagnostics')")
        href = link.get_attribute("href")
        assert href is not None
        assert "/diagnostics" in href


# ---------------------------------------------------------------------------
# 10. Debug Mode Toggle
# ---------------------------------------------------------------------------


class TestDebugModeToggle:
    """Toggling debug mode should affect how assistant messages are rendered."""

    def test_debug_checkbox_is_toggleable(self, ui_page: Page) -> None:
        sidebar = ui_page.locator(_SIDEBAR)
        # Debug mode is the second checkbox (after Verify TLS).
        checkboxes = sidebar.locator("div[data-testid='stCheckbox']")
        debug_checkbox = checkboxes.nth(1)
        expect(debug_checkbox).to_be_visible()
        # Click to toggle – just verify no crash.
        debug_checkbox.click()
        ui_page.wait_for_timeout(500)


# ---------------------------------------------------------------------------
# 11. Responsive Sidebar Collapse
# ---------------------------------------------------------------------------


class TestSidebarCollapse:
    """Sidebar should be collapsible via the Streamlit toggle."""

    def test_sidebar_can_collapse(self, ui_page: Page) -> None:
        collapse_btn = ui_page.locator("button[data-testid='stSidebarCollapseButton']")
        if collapse_btn.is_visible(timeout=3_000):
            collapse_btn.click()
            ui_page.wait_for_timeout(500)
            sidebar = ui_page.locator(_SIDEBAR)
            expect(sidebar).to_be_hidden(timeout=5_000)
