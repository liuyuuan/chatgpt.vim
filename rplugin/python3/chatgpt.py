import enum
import json
import os
from typing import Optional

import openai
import pynvim
from jinja2 import Environment, FileSystemLoader


class ChatType(enum.Enum):
    CHAT = 0
    MINICHAT = 1


class MiniChatType(enum.Enum):
    POPUP = 0
    REPLACE = 1
    APPEND = 2


class WorkspaceManager:
    def __init__(self):
        self.root_path = os.path.join(os.environ["HOME"], ".config", "chatgpt")
        self.config_path = os.path.join(self.root_path, "config.json")
        self.prompts_path = os.path.join(self.root_path, "prompts")
        self.chats_path = os.path.join(self.root_path, "chats")

    def setup(self):
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        if not os.path.exists(self.prompts_path):
            os.makedirs(self.prompts_path)
        if not os.path.exists(self.chats_path):
            os.makedirs(self.chats_path)
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump({"OPENAI_API_KEY": ""}, f)

    @property
    def config(self):
        return json.load(open(self.config_path))

    def get_prompt(self, command_name):
        env = Environment(loader=FileSystemLoader(self.prompts_path))
        template = env.get_template(f"{command_name}.prompt")
        return template

    def list_prompts(self):
        return [prompt.split(".")[0] for prompt in os.listdir(self.prompts_path)]

    def create_prompt(self, name, nvim):
        filename = os.path.join(self.prompts_path, f"{name}.prompt")
        example_prompt = """# This is an example prompt, delete this and write your own prompt, make sure it's a legal json
[
    {"role": "system", "content": "YOUR SYSTEM PROMPT"},
    {"role":"user", "content": "YOUR CONTENT, YOU CAN USE {{ content | safe }} TO REPRESENT SELECTED CONTENT"},
    ...
]
        """
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(example_prompt)
            # full path of the file example_prompt.json under scripts folder
        nvim.command(f"botright 20new {filename} | set ft=json")


class Chat:
    def __init__(self, nvim, winid: int, bufnr: int):
        self.nvim = nvim
        self.topic_winid = winid
        self.topic_bufnr = bufnr
        self.create()

        # windows
        self.chat_window = None
        self.input_window = None
        self.answer_window = None

    def open_window(self, buffer, focus, width, height, row, col):
        return self.nvim.api.open_win(
            buffer,
            focus,
            {
                "relative": "editor",
                "width": width,
                "height": height,
                "row": row,
                "col": col,
                "border": self.border,
                "style": "minimal",
            },
        )

    def create(self):
        self.show_width = self.nvim.options["columns"] // 2
        self.show_height = (self.nvim.options["lines"] - 3) // 3 * 2
        self.input_width = self.show_width
        self.input_height = min(self.show_height // 2, 5)
        self.border = "rounded"
        self.init_buffers()

    def init_buffers(self):
        self.chat_bufnr = self.nvim.funcs.bufadd("__show_chat__")
        self.input_bufnr = self.nvim.funcs.bufadd("__input__")
        self.answer_bufnr = self.nvim.funcs.bufadd("__answer__")

        self.nvim.api.buf_set_option(self.chat_bufnr, "buftype", "nofile")
        self.nvim.api.buf_set_option(self.input_bufnr, "buftype", "nofile")
        self.nvim.api.buf_set_option(self.answer_bufnr, "buftype", "nofile")

        self.nvim.api.buf_set_option(self.chat_bufnr, "filetype", "markdown")
        self.nvim.api.buf_set_option(self.chat_bufnr, "ts", 4)
        self.nvim.api.buf_set_option(self.chat_bufnr, "sts", 4)
        self.nvim.api.buf_set_option(self.chat_bufnr, "shiftwidth", 4)
        self.nvim.api.buf_set_option(self.chat_bufnr, "expandtab", True)
        self.nvim.api.buf_set_option(self.answer_bufnr, "filetype", "markdown")

    def show(self, show_type: ChatType, mini_chat_type=None):
        if show_type == ChatType.CHAT:
            # creates the chat window
            self.chat_window = self.open_window(
                self.chat_bufnr,
                0,
                self.show_width,
                self.show_height,
                0,
                self.show_width // 2,
            )

            # creates the input window
            self.input_window = self.open_window(
                self.input_bufnr,
                1,
                self.input_width,
                self.input_height,
                self.show_height + 2,
                self.show_width // 2,
            )
            # enter insert mode
            self.nvim.command("startinsert")
        elif show_type == ChatType.MINICHAT and mini_chat_type == MiniChatType.POPUP:
            # clean answer buffer
            self.nvim.api.buf_set_lines(self.answer_bufnr, 0, -1, 1, [])
            self.answer_window = self.open_window(
                self.answer_bufnr,
                1,
                self.show_width,
                self.show_height,
                0,
                self.show_width // 2,
            )

    @property
    def showing(self):
        return (
            (
                self.chat_window is not None
                and self.nvim.api.win_is_valid(self.chat_window.handle)
                and self.input_window is not None
                and self.nvim.api.win_is_valid(self.input_window.handle)
            )
        ) or (
            self.answer_window is not None
            and self.nvim.api.win_is_valid(self.answer_window.handle)
        )

    def hide(self):
        if self.chat_window is not None and self.nvim.api.win_is_valid(
            self.chat_window.handle
        ):
            self.nvim.api.win_close(self.chat_window.handle, 1)
            self.chat_window = None
        if self.input_window is not None and self.nvim.api.win_is_valid(
            self.input_window.handle
        ):
            self.nvim.api.win_close(self.input_window.handle, 1)
            self.input_window = None
        if self.answer_window is not None and self.nvim.api.win_is_valid(
            self.answer_window.handle
        ):
            self.nvim.api.win_close(self.answer_window.handle, 1)
            self.answer_window = None

    def input(self):
        if self.input_window is None or not self.nvim.api.win_is_valid(
            self.input_window.handle
        ):
            self.show(ChatType.CHAT)
        input_lines = self.nvim.api.buf_get_lines(self.input_bufnr, 0, -1, True)
        content = "\n".join(input_lines)
        self.nvim.api.buf_set_lines(self.input_bufnr, 0, -1, True, [])
        return {"role": "user", "content": content}

    def safe_select(self):
        """
        Gets the selected text in the topic buffer, handles wide characters correctly

        Returns:
            text: the selected text
            start_ln: the line number of the start of the selection, in bytes
            start_col: the column number of the start of the selection, in bytes
            end_ln: the line number of the end of the selection, in bytes
            end_col: the column number of the end of the selection, in bytes
        """
        char_pos_start = self.nvim.funcs.getcharpos("'<")
        char_pos_end = self.nvim.funcs.getcharpos("'>")
        start = [c - 1 for c in char_pos_start[1:3]]
        end = [c - 1 for c in char_pos_end[1:3]]

        lines = self.nvim.api.buf_get_lines(
            self.topic_bufnr, start[0], end[0] + 1, True
        )
        if len(lines) == 0:
            return "", 0, 0, 0, 0
        if len(lines) == 1:
            text = lines[0][start[1] : end[1] + 1]
        else:
            lines[0] = lines[0][start[1] :]
            lines[-1] = lines[-1][: end[1] + 1]
            text = "\n".join(lines)

        # pos_end in bytes might be wrong if the last character is a wide character
        # so we try to move the cursor right and then get the position again
        self.nvim.funcs.setcursorcharpos(end[0] + 1, end[1] + 2)
        is_last_char = self.nvim.funcs.getcharpos(".")[2] == end[1] + 1
        ln_end, col_end = self.nvim.funcs.getpos("'>")[1:3]
        ln_start, col_start = self.nvim.funcs.getpos("'<")[1:3]
        if is_last_char:
            col_end = self.nvim.funcs.col("$")
        # move cursor back to the original position anyway
        self.nvim.funcs.setcursorcharpos(end[0] + 1, end[1] + 1)

        # if use 'V', the end is bigger than col('$')
        col_end = min(col_end, self.nvim.funcs.col("$"))

        return text, ln_start, col_start, ln_end, col_end

    # writes a chunk of response to corresponding buffer
    def write_chunk(self, chunk, chat_type: ChatType, mini_chat_type=None):

        start = "role" in chunk["choices"][0]["delta"]
        if start:
            # initialize
            if chat_type == ChatType.MINICHAT and mini_chat_type == MiniChatType.POPUP:
                assert self.answer_window is not None and self.answer_window.valid
                target_winid = self.answer_window.handle
            elif (
                chat_type == ChatType.MINICHAT and mini_chat_type == MiniChatType.APPEND
            ):
                target_winid = self.topic_winid
            elif (
                chat_type == ChatType.MINICHAT
                and mini_chat_type == MiniChatType.REPLACE
            ):
                target_winid = self.topic_winid
            else:
                assert self.chat_window is not None and self.chat_window.valid
                target_winid = self.chat_window.handle

            self.nvim.funcs.win_gotoid(target_winid)

            # set cursor according to mini_chat_type
            if chat_type == ChatType.CHAT:
                self.nvim.command("normal! G")

            if chat_type == ChatType.MINICHAT and mini_chat_type == MiniChatType.APPEND:
                _, _, _, ln_end, col_end = self.safe_select()
                self.nvim.funcs.setcursorcharpos(ln_end, col_end)

            if (
                chat_type == ChatType.MINICHAT
                and mini_chat_type == MiniChatType.REPLACE
            ):
                # clear the selected range
                _, ln_start, col_start, ln_end, col_end = self.safe_select()
                self.nvim.api.buf_set_text(
                    self.topic_bufnr,
                    ln_start - 1,
                    col_start - 1,
                    ln_end - 1,
                    col_end - 1,
                    [],
                )

            self.nvim.command("startinsert")

        if chunk["choices"][0].get("finish_reason") != "stop":
            if start:
                text = chunk["choices"][0]["delta"].get("role")
                text = f"**{text.upper()}**: "
                if mini_chat_type in (MiniChatType.APPEND, MiniChatType.REPLACE):
                    text = ""
            else:
                text = chunk["choices"][0]["delta"].get("content")

            self.nvim.funcs.feedkeys(f"{text}", "n")
        else:
            self.nvim.command('exec "normal! o\<Esc>gI\<Esc>"')
            if self.input_window is not None and self.input_window.valid:
                self.nvim.funcs.win_gotoid(self.input_window.handle)
                self.nvim.command("startinsert")

    def send(self, request):
        # get input_bufnr content as list
        if isinstance(request, dict):
            request = [request]
        contents = []
        for req in request:
            content = f"**{req['role']}**: {req['content']}".split("\n") + [
                " " * self.show_width
            ]
            contents.extend(content)
        # always write to chat buffer
        self.nvim.api.buf_set_lines(self.chat_bufnr, -1, -1, True, contents)


@pynvim.plugin
class ChatGPTClient:
    def __init__(self, nvim):
        self.nvim = nvim
        self.workspace = WorkspaceManager()
        openai.api_key = self.workspace.config["OPENAI_API_KEY"]
        self._history = []
        self.chat = None
        self._chunk_buffer = []

    def create_chat(self):
        if self.nvim.funcs.bufname() in ("__input__", "__show_chat__", "__answer__"):
            assert self.chat is not None
            return self.chat

        topic_winid = self.nvim.funcs.win_getid()
        topic_bufnr = self.nvim.funcs.bufnr("%")

        if self.chat is None:
            self.chat = Chat(self.nvim, topic_winid, topic_bufnr)
            return self.chat

        if self.chat.topic_bufnr == topic_bufnr:
            return self.chat

        return Chat(self.nvim, topic_winid, topic_bufnr)

    @pynvim.command("ToggleChat", nargs=0)
    def toggle_chat(self):
        # create chat anyway
        self.chat = self.create_chat()

        if self.chat.showing:
            self.chat.hide()
        else:
            self.chat.show(ChatType.CHAT)

    def accept(self, chunk):
        def _merge_chunk(buffer):
            merged = []
            for c in buffer:
                if c["choices"][0]["delta"].get("role") == "assistant":
                    continue
                merged.append(c["choices"][0]["delta"]["content"])
            return {"role": "assistant", "content": "".join(merged)}

        if not chunk:
            return

        if chunk["choices"][0].get("finish_reason") == "stop":
            self._history.append(_merge_chunk(self._chunk_buffer))
            self._chunk_buffer = []
            return

        if self._chunk_buffer and chunk["id"] != self._chunk_buffer[-1]["id"]:
            self._history.append(_merge_chunk(self._chunk_buffer))
            self._chunk_buffer = [chunk]
            return

        self._chunk_buffer.append(chunk)

    def send(
        self,
        n_turn,
        message=None,
        chat_type=ChatType.CHAT,
        mini_chat_type: Optional[MiniChatType] = None,
    ):
        if self.chat is None:
            return
        if message is None:
            message = self.chat.input()
        if isinstance(message, list):
            self._history.extend(message)
        elif isinstance(message, dict):
            self._history.append(message)
        else:
            raise ValueError("message must be list or dict")
        self.chat.send(message)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=self._history[-int(n_turn) :],
        )

        for chunk in response:
            self.accept(chunk)
            self.chat.write_chunk(chunk, chat_type, mini_chat_type=mini_chat_type)

    @pynvim.command("ChatGPTSend", nargs=1)
    def chat_send(self, args):
        self.send(int(args[0]))

    def minichat_send(self, message, mini_chat_type):
        if self.chat is not None and mini_chat_type == MiniChatType.POPUP:
            self.chat.show(ChatType.MINICHAT, mini_chat_type=mini_chat_type)
        self.send(
            len(message),
            message=message,
            chat_type=ChatType.MINICHAT,
            mini_chat_type=mini_chat_type,
        )

    @pynvim.function("MiniChat", range=True)
    def mini_chat(self, args, _):
        # create chat anyway
        self.chat = self.create_chat()

        cmd_name, write_type = args
        select = self.chat.safe_select()
        content = select[0]

        template = self.workspace.get_prompt(cmd_name)
        # TODO: a bit tricky here, try to fix this hack later
        prompt = template.render(content=json.dumps(content)[1:-1])
        message = json.loads(prompt)
        self.minichat_send(message, MiniChatType[write_type.upper()])

    @pynvim.command("CreatePrompt", nargs=1)
    def create_prompt(self, args):
        prompt_name = args[0]
        self.workspace.create_prompt(prompt_name, self.nvim)
        self.create_prompt_commands(prompt_name)

    @pynvim.command("ChatGPTSetup", nargs=0)
    def setup(self):
        self.workspace.setup()

    def create_prompt_commands(self, prompt_name=None):
        if prompt_name is not None:
            self.nvim.command(
                "command! -nargs=1 -range -complete=customlist,ChatPromptComplete"
                + f" Prompt{prompt_name} :call MiniChat('{prompt_name}', <f-args>)"
            )
        else:
            for prompt_name in self.workspace.list_prompts():
                self.nvim.command(
                    "command! -nargs=1 -range -complete=customlist,ChatPromptComplete"
                    + f" Prompt{prompt_name} :call MiniChat('{prompt_name}', <f-args>)"
                )

    @pynvim.autocmd("VimEnter", pattern="*", eval='expand("<afile>")', sync=True)
    def on_vimenter(self, _):
        self.create_prompt_commands()
