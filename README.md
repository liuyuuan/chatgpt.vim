# ChatGPT in neovim

This is a chatgpt client for neovim. You can seamlessly employ ChatGPT within
Neovim, engaging in a dialogue with the model or executing commands through
customized prompts.

This plugin utilizes remote plugins of Neovim, therefore during the installation
process, it is imperative to manually execute the command `:UpdateRemotePlugins`
in order to update the configuration of remote plugins.

## Basic Usage

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
