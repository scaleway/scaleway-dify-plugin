# Import dify_plugin first, as main.py does: it patches networking (gevent)
# before requests/urllib3 are loaded, which avoids gevent's warning.
import dify_plugin  # noqa: F401
