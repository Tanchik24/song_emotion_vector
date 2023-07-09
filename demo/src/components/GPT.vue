<template>
  <div class="gpt">
    <div class="gpt__text-container">
      <p class="gpt__text">Основываясь на этих значениях, эмоции песни будут отрицательными и
        низкими по
        степени
        возбуждения. Его можно охарактеризовать как печальный или меланхоличный, с очень низким
        уровнем энергии. Скорее всего, это не будет песня, которая будет мотивировать или заряжать
        слушателя энергией, но вместо этого может вызвать чувство размышления или созерцания.</p>
    </div>
  </div>
</template>
<style lang="scss">
.gpt {
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background-image: linear-gradient(
      to bottom,
      rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
  );
  display: flex;
  justify-content: center;
  align-content: center;

  &__text-container {
    width: 80%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  &__text {
    text-align: left;
  }
}
</style>
<script lang="ts">
</script>
