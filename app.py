import streamlit as st

from crypto_llm.chainer import LlmChainer
from crypto_llm.storage import FileStorage

from dotenv import load_dotenv

load_dotenv("./.env")


@st.cache_data
def initialize_storage():
    # Создаем экземпляр вашего класса LlmChainer
    chainer = LlmChainer()

    # Загружаем список криптовалют
    storage = FileStorage()
    # storage.get_all_cmc_list()
    storage.save_cmc_info()
    crypto_list = storage.show_all_currency_names()

    return chainer, storage, crypto_list


# Инициализируем данные
chainer, storage, crypto_list = initialize_storage()

# Заголовок приложения
st.title("Crypto Q&A")

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
if st.button("Получить summary по криптовалюте"):
    # Получаем резюме
    summary = chainer.run_chain(selected_crypto, is_summary=True)
    st.write("Summary:", summary)
