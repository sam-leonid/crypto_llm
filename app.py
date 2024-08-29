import streamlit as st

from crypto_llm.chainer import LlmChainer
from crypto_llm.loader import CMCLoader
from dotenv import load_dotenv

load_dotenv("./.env")

# Предполагаем, что у вас есть доступ к классам LlmChainer и CMCLoader
# from your_module import LlmChainer, CMCLoader

# Создаем экземпляр вашего класса LlmChainer
chainer = LlmChainer()

# Загружаем список криптовалют
cmc_loader = CMCLoader()
crypto_list = cmc_loader.show_all_currency_names()

# Заголовок приложения
st.title("Crypto Q&A with LlmChainer")

# Поле для выбора криптовалюты
selected_crypto = st.selectbox("Выберите криптовалюту:", crypto_list)

# Поле для ввода вопроса
question = st.text_input("Введите ваш вопрос по криптовалюте:")

# Кнопка для получения ответа
if st.button("Получить ответ"):
    if question:
        # Получаем ответ по введенному вопросу
        answer = chainer.run_chain(selected_crypto, question)
        st.write("Ответ:", answer)
    else:
        st.warning("Пожалуйста, введите вопрос.")

# Кнопка для получения резюме
if st.button("Получить резюме"):
    # Получаем резюме
    summary = chainer.run_chain(selected_crypto, is_summary=True)
    st.write("Резюме:", summary)

# chainer.run_chain('Solana', "Какой алгоритм консенсуса в SOLANA?")
# chainer.run_chain("ChatCoin", is_summary=True)
# chainer.run_chain("Lition", is_summary=True)
