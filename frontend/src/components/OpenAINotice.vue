<script setup lang="ts">
import AppIcon from './AppIcon.vue'
</script>

<template>
  <div class="notice notice-warn" role="status">
    <AppIcon name="alert" />
    <div class="notice-body">
      <p><strong>OpenAI is not set up yet, so new clips cannot be generated.</strong></p>
      <p>
        Add <code>OPENAI_API_KEY=your-key</code> to the <code>.env</code> file in the project folder, then restart
        Local Clipper. The key stays on this machine; this page never asks for it.
      </p>
    </div>
  </div>
</template>
