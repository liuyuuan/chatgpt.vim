# ChatGPT in neovim

This is a chatgpt client for neovim. You can seamlessly employ ChatGPT within
Neovim, engaging in a dialogue with the model or executing commands through
customized prompts.

## Install
Clone this repo into your `runtimepath`, or use Vundle.
This is a remote plugin, remember to call `:UpdateRemotePlugin` at first.

## Basic Usage

<video src="https://user-images.githubusercontent.com/1546064/229753659-d02ae1ab-b5af-4e69-9c80-d367b80ee92f.mov" controls="controls" style="max-width: 730px;" ></video>

- commands:

    - `:ChatGPTSetup`: Initializes the plugin
    - `:ToggleChat`:Open the dialogue box and initiate a conversation with GPT-3.5-Turbo 
    - `:ChatGPTSend ${n}`:Send the most recent n dialogues to the model. 
    - `:CreatePrompt ${YOUR_PROMPT_NAME}`: Create a prompt for a task

- recommand key maps:

```vim
nnoremap <leader>c :ToggleChat<CR>
nnoremap <leader>s :ChatGPTSend 1<CR>
nnoremap <leader>a :ChatGPTSend 10<CR>
```
