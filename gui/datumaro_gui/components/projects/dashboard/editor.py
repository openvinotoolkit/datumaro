from functools import partial
from streamlit_elements import mui, editor, sync, lazy
from .dashboard import Dashboard


class Editor(Dashboard.Item):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dark_theme = False
        self._index = 0
        self._tabs = {}
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider"
        }

    def _change_tab(self, _, index):
        self._index = index

    def update_content(self, label, content):
        self._tabs[label]["content"] = content

    def add_tab(self, label, default_content, language):
        self._tabs[label] = {
            "content": default_content,
            "language": language
        }

    def get_content(self, label):
        return self._tabs[label]["content"]

    def __call__(self):
        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):

            with self.title_bar("0px 15px 0px 15px"):
                mui.icon.Terminal()
                mui.Typography("Editor")

                with mui.Tabs(value=self._index, onChange=self._change_tab, scrollButtons=True, variant="scrollable", sx={"flex": 1}):
                    for label in self._tabs.keys():
                        mui.Tab(label=label)

            for index, (label, tab) in enumerate(self._tabs.items()):
                with mui.Box(sx=self._editor_box_style, hidden=(index != self._index)):
                    editor.Monaco(
                        css={"padding": "0 2px 0 2px"},
                        defaultValue=tab["content"],
                        language=tab["language"],
                        onChange=lazy(partial(self.update_content, label)),
                        theme="vs-dark" if self._dark_mode else "light",
                        path=label,
                        options={
                            "wordWrap": True
                        }
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                mui.Button("Apply", variant="contained", onClick=sync())
                mui.Typography("Or press ctrl+s", sx={"flex": 1})
