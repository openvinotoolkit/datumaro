from streamlit_elements import media, mui, sync, lazy
from .dashboard import Dashboard

class Player(Dashboard.Item):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._url = "https://www.youtube.com/watch?v=CmSKVW1v0xM"

    def _set_url(self, event):
        self._url = event.target.value

    def __call__(self):
        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                mui.icon.OndemandVideo()
                mui.Typography("Media player")

            with mui.Stack(direction="row", spacing=2, justifyContent="space-evenly", alignItems="center", sx={"padding": "10px"}):
                mui.TextField(defaultValue=self._url, label="URL", variant="standard", sx={"flex": 0.97}, onChange=lazy(self._set_url))
                mui.IconButton(mui.icon.PlayCircleFilled, onClick=sync(), sx={"color": "primary.main"})

            media.Player(self._url, controls=True, width="100%", height="100%")
