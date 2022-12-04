# hse_ml_hw1
Проведён анализ среза данных об автомобилях, применимости регрессионных моделей в задаче предсказания цены на автомобиль в раззрезе указанных данных и построен микросервис для получения предсказаний с помощью FastAPI
1)Быстрая EDA по основным интересующим моментам, что на личном опыте будет важнее далее для feauture engineering
2)Данные очищены подправлены
3)Построен пайплайн для дополнительной обработки данных и обучения рег. модели
4)Модель затюнена до 0.91 R^2 и 0.345 бизнесовой метрики (которая показывает долю предсказаний улетевших от факта меньше чем на 10%)
5)Все характеристики модели для дальнейшего инференса запиклены и перенесены далее
6)Построен сервис на FastAPI получающий на вход либо JSON запись одной машины (одно наблюдение) и выдающий в ответ предсказанную цену с помощью модели из пикла. Так же есть возможность заливать csv файл с машинами (нужного формата) и получать в ответ csv файл с добавленным столбцом прогнозных цен


Что дало буст и что сделано: Перепробовал всё, приводил к нормальному фичи, менял их на другие функции, добавлял новые фичи, трансформил всё, юзал робастные лин модели и тд и тп....В итоге ничего не дало повышения бизнес метрики (но R^2 повышался) кроме получения столбца производителя и преобразования Y в np.log(Y)

При этом очень задумался в начальном этапе и построил архитектуру так, что разные импутеры тупо не протестить. Дедлайн жмёт - перекраивать всё не хотелось, но с учётом того, как мы заполняли и КАКИЕ ИМЕННО пропуски - возможно групповые импутеры бы пригодились очень очень хорошо с учётом специфики машин и их моделей и их лет. Ещё возможно сетки можно было пошире делать, но времени ждать не было...

Итого считаю что бэггинги и бустинги конечно могут сильно здесь переобучиться, но данные по машинам вполне конечные, поэтому при должной усидчивости во время работы скраппера, можно сделать вполне крутую модель. Обычная линейка даёт очень хорошие резы без идеального тюнинга.
