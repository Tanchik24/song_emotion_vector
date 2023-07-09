import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import HomeView from '../views/HomeView.vue';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/info',
    name: 'info',
    component: () => import('../views/MusicView.vue'),
  },
  {
    path: '/music',
    name: 'music',
    component: () => import('../views/MusicInfoView.vue'),
  },
  {
    path: '/musicinfo',
    name: 'musicinfo',
    component: () => import('../views/MusicInfoViewMobileView.vue'),
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
