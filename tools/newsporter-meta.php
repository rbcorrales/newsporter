<?php
/**
 * Plugin Name: Newsporter source-id meta
 * Description: Registers `_newsporter_source_id` and `_newsporter_byline` post meta with REST visibility, so newsporter can look up posts by their source-corpus id and skip re-creating them on retry / re-run.
 * Author: Newsporter
 * Version: 0.2.0
 *
 * Drop in `wp-content/mu-plugins/` on the target WordPress site.
 *
 * Without this, WP silently drops both meta keys on every POST and
 * newsporter's idempotency lookup never finds existing posts, so a
 * re-run will duplicate every post.
 */

if (!defined('ABSPATH')) {
    exit;
}

add_action('init', function () {
    // `manage_options` (administrators) only. These are infrastructure
    // meta keys, not author content; a contributor must not be able to
    // hijack newsporter's idempotency map by writing arbitrary
    // _newsporter_source_id values onto posts they can edit.
    $auth = function ($allowed, $meta_key, $object_id, $user_id) {
        return user_can($user_id, 'manage_options');
    };
    register_post_meta('post', '_newsporter_source_id', [
        'type'              => 'string',
        'description'       => 'Stable id from the source corpus (the row id Newsporter pulled from the dataset).',
        'single'            => true,
        'show_in_rest'      => true,
        'auth_callback'     => $auth,
        'sanitize_callback' => 'sanitize_text_field',
    ]);
    register_post_meta('post', '_newsporter_byline', [
        'type'              => 'string',
        'description'       => 'Synthesized author name (display only).',
        'single'            => true,
        'show_in_rest'      => true,
        'auth_callback'     => $auth,
        'sanitize_callback' => 'sanitize_text_field',
    ]);
});
