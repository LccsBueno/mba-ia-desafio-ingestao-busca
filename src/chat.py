from search import search_prompt

def main():
    
    while True:
        entrada = input("PERGUNTA: ")
        
        if entrada == "":
            print("Encerrando conversa")
            break
        
        chain = search_prompt(entrada)

        if not chain:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return
        
        print("RESPOSTA: "+chain.content)
        # pass

if __name__ == "__main__":
    main()